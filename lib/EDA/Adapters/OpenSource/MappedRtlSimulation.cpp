#include "MappedRtlSimulationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/MappedRtlSimulation.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::eda::open_source {
namespace {

using namespace evaluation;
using namespace external_tool;

constexpr ModelOutputSlotRef kExecutionOutput(0);

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_simulation_invalid: " + detail);
}

llvm::Error rejectUndeclaredOutputs(llvm::StringRef bundleRoot) {
  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  const std::set<std::string> expected{
      "completion.json",
      std::filesystem::path(mappedRtlResultPath.str()).filename().string(),
      "stderr.log", "stdout.log"};
  std::set<std::string> found;
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return invalid("outputs is not an ordinary directory");
  for (std::filesystem::directory_iterator iterator(outputs, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string name = path.filename().string();
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) || !expected.count(name))
      return invalid("outputs contains undeclared entry '" + name + "'");
    found.insert(name);
  }
  if (error)
    return invalid("could not enumerate outputs: " + error.message());
  if (found != expected)
    return invalid("outputs omits a lifecycle or declared result file");
  return llvm::Error::success();
}

EvaluationModelResult terminalResult(EvaluationEvidenceOutcome outcome) {
  return EvaluationModelResult{{{kExecutionOutput, {}}}, std::move(outcome)};
}

llvm::Expected<EvaluationModelResult>
classifyFailedAttempt(const FailedExternalToolInvocationAttempt &failed) {
  switch (failed.status) {
  case InvocationCompletionStatus::Success:
    return invalid("failed invocation outcome carries success status");
  case InvocationCompletionStatus::MissingEnvironment:
  case InvocationCompletionStatus::ModuleActivationFailed:
  case InvocationCompletionStatus::VersionMismatch:
    return terminalResult(
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable});
  case InvocationCompletionStatus::BundleContentMismatch:
    return invalid("invocation bundle content changed before execution");
  case InvocationCompletionStatus::ToolExit:
  case InvocationCompletionStatus::MissingOutput:
    return terminalResult(ExecutionFailedEvidence{OutcomeReason::ToolFailure});
  }
  llvm_unreachable("closed invocation status");
}

llvm::Expected<MappedRtlExecutionClosure>
deriveExecutionClosure(const EvaluationRequest &request) {
  auto configuration =
      evaluation::models::projectVerifiedMappedRtlSimulationConfiguration(
          request);
  if (!configuration)
    return configuration.takeError();
  const auto implementations = request.subjectBindings().subjects(
      evaluation::models::mappedRtlHardwareImplementationSubjectRole());
  const auto deployments = request.subjectBindings().subjects(
      evaluation::models::mappedRtlDeploymentSubjectRole());
  if (implementations.size() != 1 || deployments.size() != 1 ||
      !request.workload() || !request.runtimeInput())
    return invalid("Request does not bind one complete mapped RTL case");
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  return MappedRtlExecutionClosure{configuration->providerBinding,
                                   std::move(*contract),
                                   implementations.front(),
                                   deployments.front(),
                                   *request.workload(),
                                   *request.runtimeInput()};
}

llvm::Expected<EvaluationModelProviderPreparation>
prepareProvider(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const ExternalToolPreparationContext &context) {
  (void)resolution;
  auto closure = deriveExecutionClosure(request);
  if (!closure)
    return closure.takeError();

  const ExternalToolProviderDescriptor &toolProvider = verilatorProvider();
  auto options = resolveMappedRtlExecutionAttemptOptions(context.localConfig);
  if (!options)
    return options.takeError();

  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe toolProbe(probeRoot.string(),
                                  toolProvider.versionProbe);
  auto tool = resolveToolBinding(toolProvider.binding, context.localConfig,
                                 captureToolEnvironment(toolProvider.binding),
                                 toolProbe);
  if (!tool)
    return tool.takeError();
  if (tool->version !=
      closure->simulatorBinding.stableHdlSimulatorBuildIdentity)
    return invalid("resolved Verilator build differs from the model binding");
  if (!findValidatedRelease(toolProvider.binding.key, tool->version))
    return EvaluationModelProviderPreparation{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<std::string> inheritEnvironment = options->inheritedEnvironment;
  const ExternalToolProviderDescriptor &containerProvider =
      polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       containerProvider.versionProbe);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, containerProvider.binding,
      captureToolEnvironment(containerProvider.binding), containerProbe,
      toolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &container,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(probeRoot.string(), resolvedTool,
                                             toolProvider.versionProbe,
                                             container, os, inheritEnvironment);
      });
  if (!runtime)
    return runtime.takeError();

  auto buildTools = resolveMappedRtlBuildTools(context.localConfig);
  if (!buildTools)
    return buildTools.takeError();
  auto projection = deriveMappedRtlExecutionBundleProjection(
      *closure,
      MappedRtlVerilationPlan{options->cycleLimit, options->buildJobs,
                              options->modelThreads, tool->executable,
                              *buildTools},
      artifacts, blobs);
  if (!projection)
    return projection.takeError();
  if (const auto *unsupported = std::get_if<UnsupportedEvidence>(&*projection))
    return EvaluationModelProviderPreparation{*unsupported};
  auto bundle =
      std::get<MappedRtlExecutionBundleProjection>(std::move(*projection));
  std::vector<MaterializedBundleFile> files{
      {bundle.testbenchPath, std::move(bundle.testbench), std::nullopt, false},
      {bundle.standaloneVerilatorDriverPath,
       std::move(bundle.standaloneVerilatorDriver), std::nullopt, false}};
  files.insert(files.end(),
               std::make_move_iterator(bundle.toolLocalInputs.begin()),
               std::make_move_iterator(bundle.toolLocalInputs.end()));
  files.insert(files.end(),
               std::make_move_iterator(bundle.semanticInputs.begin()),
               std::make_move_iterator(bundle.semanticInputs.end()));

  const std::string executable = tool->executable;
  ExternalToolInvocationBundleSpec specification{
      closure->semanticContract,
      std::move(*tool),
      toolProvider.versionProbe,
      std::move(*runtime),
      containerProvider.versionProbe,
      {{executable, "-f", bundle.standaloneVerilatorDriverPath},
       std::move(bundle.buildCommand),
       {bundle.simulatorExecutablePath}},
      std::move(inheritEnvironment),
      {bundle.resultPath},
      std::move(files),
      {},
      {},
      {bundle.simulatorExecutablePath}};
  specification.diagnosticCommandOrdinals = {2};
  specification.auxiliaryToolExecutables =
      std::move(buildTools->provenance);
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<EvaluationModelResult> importProviderImpl(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolInvocationExecutionObservation *executionObservation =
        nullptr) {
  auto closure = deriveExecutionClosure(request);
  if (!closure)
    return closure.takeError();
  auto expectation =
      deriveMappedRtlExecutionImportExpectation(*closure, artifacts, blobs);
  if (!expectation)
    return expectation.takeError();
  auto attempt =
      executionObservation
          ? importExternalToolInvocationAttempt(prepared, *expectation,
                                                *executionObservation)
          : importExternalToolInvocationAttempt(prepared, *expectation);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt))
    return classifyFailedAttempt(*failed);

  ImportedExternalToolInvocationBundle imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  if (llvm::Error error = rejectUndeclaredOutputs(prepared.bundleRoot))
    return std::move(error);
  auto resultBytes =
      readExternalToolInvocationDeclaredOutput(imported, mappedRtlResultPath);
  if (!resultBytes)
    return resultBytes.takeError();
  auto result = parseMappedRtlSimulationResult(*resultBytes);
  if (!result)
    return result.takeError();
  if (result->terminal == MappedRtlTerminalStatus::StoppedByLimit)
    return terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
  if (!result->retirementCycle ||
      result->launchCycle > *result->retirementCycle ||
      *result->retirementCycle > result->terminalCycle)
    return invalid("retired result has inconsistent progress coordinates");
  const std::uint64_t cycleCount =
      *result->retirementCycle - result->launchCycle;
  if (cycleCount >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return terminalResult(
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure});

  auto boundary = projectMappedRtlSpatialEngineBoundaryResult(*closure, *result,
                                                              artifacts, blobs);
  if (!boundary)
    return boundary.takeError();

  sim::SpatialSimulationExecution execution{
      evaluationRequestReference(request), std::move(boundary->terminal),
      std::move(boundary->functionalObservations),
      std::move(boundary->progressObservations),
      std::move(boundary->activitySummaries)};
  auto finalized =
      sim::finalizeSimulationExecution(execution, resolution, artifacts, blobs);
  if (!finalized)
    return finalized.takeError();
  auto reference = sim::publishSimulationExecution(*finalized, artifacts);
  if (!reference)
    return reference.takeError();

  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::CycleCount)
      return invalid("Request contains an unsupported metric");
    metrics.push_back(MetricResult{
        UncertaintyKind::ExactWithinModel,
        PointObservation{IntegerValue(static_cast<std::int64_t>(cycleCount))},
        {}});
  }
  return EvaluationModelResult{{{kExecutionOutput, {std::move(*reference)}}},
                               CompletedEvidence{std::move(metrics), {}}};
}

llvm::Expected<EvaluationModelResult>
importProvider(const EvaluationRequest &request,
               const CaseArtifactResolution &resolution,
               const PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderImpl(request, resolution, prepared, artifacts, blobs);
}

llvm::Expected<EvaluationModelResult> importProviderWithExecution(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderImpl(request, resolution, prepared, artifacts, blobs,
                            &execution);
}

} // namespace

llvm::Error registerMappedRtlSimulationProvider() {
  if (llvm::Error error =
          evaluation::models::registerMappedRtlSimulationModel())
    return error;
  static const EvaluationModelProvider provider{
      evaluation::models::mappedRtlSimulatorModelDescriptorRef(),
      EvaluationModelExternalPrepareImportProvider{
          &prepareProvider, &importProvider, nullptr,
          &importProviderWithExecution}};
  return registerEvaluationModelProvider(provider);
}

} // namespace loom::eda::open_source

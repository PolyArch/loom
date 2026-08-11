#include "MappedRtlSimulationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelProvider.h"
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
constexpr std::uint64_t kDefaultCycleLimit = 1'000'000;
constexpr std::uint64_t kDefaultBuildJobs = 4;
constexpr std::uint64_t kMaximumBuildJobs = 256;
constexpr std::uint64_t kMaximumDebugVerbosity = 3;

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_simulation_invalid: " + detail);
}

llvm::Expected<std::uint64_t>
resolveCycleLimit(const LocalToolConfig &localConfig,
                  const ExternalToolProviderDescriptor &provider) {
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured == localConfig.tools.end())
    return kDefaultCycleLimit;
  const llvm::json::Value *value =
      configured->second.providerOptions.get("max_cycles");
  if (!value)
    return kDefaultCycleLimit;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed == 0)
    return invalid("verilator.provider_options.max_cycles must be a positive "
                   "unsigned integer");
  return *parsed;
}

llvm::Expected<std::uint64_t>
resolveDebugVerbosity(const LocalToolConfig &localConfig,
                      const ExternalToolProviderDescriptor &provider) {
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured == localConfig.tools.end())
    return 0;
  const llvm::json::Value *value =
      configured->second.providerOptions.get("debug_verbose");
  if (!value)
    return 0;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed > kMaximumDebugVerbosity)
    return invalid("verilator.provider_options.debug_verbose must be an "
                   "unsigned integer from zero through three");
  return *parsed;
}

llvm::Expected<std::uint64_t>
resolveBuildJobs(const LocalToolConfig &localConfig,
                 const ExternalToolProviderDescriptor &provider) {
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured == localConfig.tools.end())
    return kDefaultBuildJobs;
  const llvm::json::Value *value =
      configured->second.providerOptions.get("build_jobs");
  if (!value)
    return kDefaultBuildJobs;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed == 0 || *parsed > kMaximumBuildJobs)
    return invalid("verilator.provider_options.build_jobs must be an "
                   "unsigned integer from one through 256");
  return *parsed;
}

std::vector<std::string>
inheritedEnvironment(const LocalToolConfig &localConfig,
                     const ExternalToolProviderDescriptor &provider) {
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured == localConfig.tools.end())
    return {};
  return configured->second.inheritEnvironment;
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

llvm::Expected<EvaluationModelProviderPreparation>
prepareProvider(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const ExternalToolPreparationContext &context) {
  auto factsOrUnsupported = detail::deriveMappedRtlInvocationFacts(
      request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<UnsupportedEvidence>(&*factsOrUnsupported))
    return EvaluationModelProviderPreparation{*unsupported};
  auto facts = std::get<detail::MappedRtlInvocationFacts>(
      std::move(*factsOrUnsupported));

  const ExternalToolProviderDescriptor &toolProvider = verilatorProvider();
  auto cycleLimit = resolveCycleLimit(context.localConfig, toolProvider);
  if (!cycleLimit)
    return cycleLimit.takeError();
  facts.cycleLimit = *cycleLimit;
  auto debugVerbosity =
      resolveDebugVerbosity(context.localConfig, toolProvider);
  if (!debugVerbosity)
    return debugVerbosity.takeError();
  auto buildJobs = resolveBuildJobs(context.localConfig, toolProvider);
  if (!buildJobs)
    return buildJobs.takeError();

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
      facts.configuration.providerBinding.stableHdlSimulatorBuildIdentity)
    return invalid("resolved Verilator build differs from the model binding");

  std::vector<std::string> inheritEnvironment =
      inheritedEnvironment(context.localConfig, toolProvider);
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

  auto testbench = detail::renderMappedRtlTestbench(facts);
  if (!testbench)
    return testbench.takeError();
  auto driver = detail::renderMappedRtlVerilatorDriver(facts, *buildJobs);
  if (!driver)
    return driver.takeError();
  std::vector<MaterializedBundleFile> files{
      {mappedRtlTestbenchPath.str(), std::move(*testbench), std::nullopt, false},
      {mappedRtlVerilatorDriverPath.str(), std::move(*driver), std::nullopt,
       false}};
  files.insert(files.end(),
               std::make_move_iterator(facts.semanticInputs.begin()),
               std::make_move_iterator(facts.semanticInputs.end()));

  const std::string executable = tool->executable;
  std::vector<std::string> simulationCommand{
      mappedRtlSimulatorExecutablePath.str()};
  if (*debugVerbosity != 0)
    simulationCommand.push_back("+LOOM_DEBUG_VERBOSE=" +
                                std::to_string(*debugVerbosity));
  ExternalToolInvocationBundleSpec specification{
      facts.semanticContract,
      std::move(*tool),
      toolProvider.versionProbe,
      std::move(*runtime),
      containerProvider.versionProbe,
      {{executable, "-f", mappedRtlVerilatorDriverPath.str()},
       std::move(simulationCommand)},
      std::move(inheritEnvironment),
      {mappedRtlResultPath.str()},
      std::move(files),
      {},
      {},
      {mappedRtlSimulatorExecutablePath.str()}};
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<EvaluationModelResult>
importProvider(const EvaluationRequest &request,
               const CaseArtifactResolution &resolution,
               const PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto expectation = detail::deriveMappedRtlImportExpectation(
      request, resolution, artifacts, blobs);
  if (!expectation)
    return expectation.takeError();
  auto attempt = importExternalToolInvocationAttempt(
      prepared, *expectation);
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
  auto resultBytes = readExternalToolInvocationDeclaredOutput(
      imported, mappedRtlResultPath);
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

  auto facts = detail::deriveMappedRtlObservationFacts(
      request, resolution, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto functional =
      detail::projectMappedRtlFunctionalObservations(*facts, *result);
  if (!functional)
    return functional.takeError();
  auto launch = ExactRatio::get(result->launchCycle, 1);
  auto retirement = ExactRatio::get(*result->retirementCycle, 1);
  auto terminal = ExactRatio::get(result->terminalCycle, 1);
  if (!launch)
    return launch.takeError();
  if (!retirement)
    return retirement.takeError();
  if (!terminal)
    return terminal.takeError();

  sim::SpatialSimulationExecution execution{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      std::move(*functional),
      sim::SpatialProgressObservations{
          sim::SpatialEventCoordinate{std::move(*launch), 0},
          sim::SpatialEventCoordinate{std::move(*retirement), 0},
          sim::SpatialEventCoordinate{std::move(*terminal), 0}},
      {}};
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

} // namespace

llvm::Error registerMappedRtlSimulationProvider() {
  if (llvm::Error error =
          evaluation::models::registerMappedRtlSimulationModel())
    return error;
  static const EvaluationModelProvider provider{
      evaluation::models::mappedRtlSimulatorModelDescriptorRef(),
      EvaluationModelExternalPrepareImportProvider{&prepareProvider,
                                                   &importProvider}};
  return registerEvaluationModelProvider(provider);
}

} // namespace loom::eda::open_source

#include "MappedRtlSimulationInternal.h"

#include "ExternalTool/Provider.h"

#include <limits>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::eda::open_source {
namespace {

constexpr std::uint64_t kDefaultCycleLimit = 1'000'000;
constexpr std::uint64_t kDefaultBuildJobs = 4;
constexpr std::uint64_t kMaximumBuildJobs = 256;
constexpr std::uint64_t kMaximumDebugVerbosity = 3;

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_execution_invalid: " + detail);
}

llvm::Expected<std::uint64_t> positiveOption(
    const external_tool::LocalToolConfig &config, llvm::StringRef name,
    std::uint64_t defaultValue, std::uint64_t maximum) {
  const auto &provider = external_tool::verilatorProvider();
  const auto configured = config.tools.find(provider.binding.key);
  if (configured == config.tools.end())
    return defaultValue;
  const llvm::json::Value *value =
      configured->second.providerOptions.get(name);
  if (!value)
    return defaultValue;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed == 0 || *parsed > maximum)
    return invalid("verilator.provider_options." + name +
                   " is outside its positive unsigned range");
  return *parsed;
}

} // namespace

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig) {
  auto cycleLimit = positiveOption(localConfig, "max_cycles",
                                   kDefaultCycleLimit,
                                   std::numeric_limits<std::uint64_t>::max());
  auto buildJobs = positiveOption(localConfig, "build_jobs", kDefaultBuildJobs,
                                  kMaximumBuildJobs);
  if (!cycleLimit || !buildJobs)
    return llvm::joinErrors(cycleLimit ? llvm::Error::success()
                                       : cycleLimit.takeError(),
                            buildJobs ? llvm::Error::success()
                                      : buildJobs.takeError());
  std::uint64_t verbosity = 0;
  const auto &provider = external_tool::verilatorProvider();
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured != localConfig.tools.end()) {
    if (const llvm::json::Value *value =
            configured->second.providerOptions.get("debug_verbose")) {
      const std::optional<std::uint64_t> parsed = value->getAsUINT64();
      if (!parsed || *parsed > kMaximumDebugVerbosity)
        return invalid("verilator.provider_options.debug_verbose must be an "
                       "unsigned integer from zero through three");
      verbosity = *parsed;
    }
  }
  return MappedRtlExecutionAttemptOptions{
      *cycleLimit, *buildJobs, verbosity,
      configured == localConfig.tools.end()
          ? std::vector<std::string>{}
          : configured->second.inheritEnvironment};
}

llvm::Expected<MappedRtlExecutionProjectionOrUnsupported>
deriveMappedRtlExecutionBundleProjection(
    const MappedRtlExecutionClosure &closure, std::uint64_t cycleLimit,
    std::uint64_t buildJobs, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (cycleLimit == 0 || buildJobs == 0)
    return invalid("execution limits must be positive");
  auto factsOrUnsupported =
      detail::deriveMappedRtlInvocationFacts(closure, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<evaluation::UnsupportedEvidence>(&*factsOrUnsupported))
    return MappedRtlExecutionProjectionOrUnsupported{*unsupported};
  detail::MappedRtlInvocationFacts facts =
      std::get<detail::MappedRtlInvocationFacts>(
          std::move(*factsOrUnsupported));
  facts.cycleLimit = cycleLimit;
  auto testbench = detail::renderMappedRtlTestbench(facts);
  if (!testbench)
    return testbench.takeError();
  auto standalone = detail::renderMappedRtlVerilatorDriver(facts, buildJobs);
  if (!standalone)
    return standalone.takeError();
  auto bridged =
      detail::renderMappedRtlBridgedVerilatorDriver(facts, buildJobs);
  if (!bridged)
    return bridged.takeError();
  return MappedRtlExecutionProjectionOrUnsupported{
      MappedRtlExecutionBundleProjection{
          std::move(facts.semanticInputs), std::move(*testbench),
          std::move(*standalone), std::move(*bridged)}};
}

llvm::Expected<external_tool::ExternalToolInvocationImportExpectation>
deriveMappedRtlExecutionImportExpectation(
    const MappedRtlExecutionClosure &closure, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  return detail::deriveMappedRtlImportExpectation(closure, artifacts, blobs);
}

llvm::Expected<sim::SpatialEngineBoundaryResult>
projectMappedRtlSpatialEngineBoundaryResult(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlSimulationResult &result, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (result.terminal != MappedRtlTerminalStatus::Retired ||
      !result.retirementCycle ||
      result.launchCycle > *result.retirementCycle ||
      *result.retirementCycle > result.terminalCycle)
    return invalid("RTL result is not a consistent retired execution");
  auto facts =
      detail::deriveMappedRtlObservationFacts(closure, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto functional =
      detail::projectMappedRtlFunctionalObservations(*facts, result);
  if (!functional)
    return functional.takeError();
  auto launch = evaluation::ExactRatio::get(result.launchCycle, 1);
  auto retirement =
      evaluation::ExactRatio::get(*result.retirementCycle, 1);
  auto terminal = evaluation::ExactRatio::get(result.terminalCycle, 1);
  if (!launch)
    return launch.takeError();
  if (!retirement)
    return retirement.takeError();
  if (!terminal)
    return terminal.takeError();
  return sim::SpatialEngineBoundaryResult{
      sim::RetiredExecution{}, std::move(*functional),
      sim::SpatialProgressObservations{
          sim::SpatialEventCoordinate{std::move(*launch), 0},
          sim::SpatialEventCoordinate{std::move(*retirement), 0},
          sim::SpatialEventCoordinate{std::move(*terminal), 0}},
      {}};
}

} // namespace loom::eda::open_source

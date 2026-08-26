#include "MappedRtlSimulationInternal.h"

#include "ExternalTool/Provider.h"

#include <filesystem>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::eda::open_source {
namespace {

constexpr std::uint64_t kDefaultCycleLimit = 1'000'000;
constexpr std::uint64_t kDefaultBuildJobs = 4;
constexpr std::uint64_t kDefaultBuildWorkers = 1;
constexpr std::uint64_t kMaximumBuildJobs = 256;
constexpr std::uint64_t kMaximumBuildWorkers = 4;

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_execution_invalid: " + detail);
}

llvm::Expected<std::uint64_t>
positiveOption(const external_tool::LocalToolConfig &config,
               llvm::StringRef name, std::uint64_t defaultValue,
               std::uint64_t maximum) {
  const auto &provider = external_tool::verilatorProvider();
  const auto configured = config.tools.find(provider.binding.key);
  if (configured == config.tools.end())
    return defaultValue;
  const llvm::json::Value *value = configured->second.providerOptions.get(name);
  if (!value)
    return defaultValue;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed == 0 || *parsed > maximum)
    return invalid("verilator.provider_options." + name +
                   " is outside its positive unsigned range");
  return *parsed;
}

llvm::Expected<std::string> canonicalPathPrefix(llvm::StringRef prefix) {
  if (prefix.empty())
    return std::string();
  if (!prefix.ends_with("/") || prefix.contains('\\') || prefix.contains('\0'))
    return invalid("bundle path prefix is not canonical");
  const std::filesystem::path path(prefix.drop_back().str());
  if (path.empty() || path.is_absolute() || path.lexically_normal() != path ||
      path == ".")
    return invalid("bundle path prefix is not canonical");
  return prefix.str();
}

llvm::Expected<std::string>
namespacedBundlePath(llvm::StringRef path, llvm::StringRef pathNamespace) {
  if (pathNamespace.empty())
    return path.str();
  if (path.contains('\\') || path.contains('\0'))
    return invalid("bundle path is not canonical");
  const std::size_t separator = path.find('/');
  if (separator == llvm::StringRef::npos || separator == 0 ||
      separator + 1 == path.size())
    return invalid("bundle path has no role-relative leaf");
  const llvm::StringRef role = path.take_front(separator);
  if (role != "drivers" && role != "inputs" && role != "outputs" &&
      role != "work")
    return invalid("bundle path has an unknown role root");
  std::string result = role.str();
  result.push_back('/');
  result += pathNamespace.str();
  result += path.drop_front(separator + 1).str();
  const std::filesystem::path normalized(result);
  if (normalized.is_absolute() || normalized.lexically_normal() != normalized)
    return invalid("namespaced bundle path is not canonical");
  return result;
}

} // namespace

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig) {
  const auto &provider = external_tool::verilatorProvider();
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured != localConfig.tools.end()) {
    for (const auto &option : configured->second.providerOptions) {
      if (option.first != "max_cycles" && option.first != "build_jobs" &&
          option.first != "build_workers")
        return invalid(
            llvm::Twine("verilator.provider_options contains unknown field ") +
            option.first.str());
    }
  }
  auto cycleLimit =
      positiveOption(localConfig, "max_cycles", kDefaultCycleLimit,
                     std::numeric_limits<std::uint64_t>::max());
  auto buildJobs = positiveOption(localConfig, "build_jobs", kDefaultBuildJobs,
                                  kMaximumBuildJobs);
  auto buildWorkers = positiveOption(
      localConfig, "build_workers", kDefaultBuildWorkers, kMaximumBuildWorkers);
  if (!cycleLimit)
    return cycleLimit.takeError();
  if (!buildJobs)
    return buildJobs.takeError();
  if (!buildWorkers)
    return buildWorkers.takeError();
  return MappedRtlExecutionAttemptOptions{
      *cycleLimit, *buildJobs, *buildWorkers,
      configured == localConfig.tools.end()
          ? std::vector<std::string>{}
          : configured->second.inheritEnvironment};
}

llvm::Expected<MappedRtlExecutionProjectionOrUnsupported>
deriveMappedRtlExecutionBundleProjection(
    const MappedRtlExecutionClosure &closure, std::uint64_t cycleLimit,
    std::uint64_t buildJobs, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix) {
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
  auto prefix = canonicalPathPrefix(pathPrefix);
  if (!prefix)
    return prefix.takeError();
  for (external_tool::MaterializedBundleFile &file : facts.semanticInputs) {
    auto path = namespacedBundlePath(file.relativePath, *prefix);
    if (!path)
      return path.takeError();
    file.relativePath = std::move(*path);
  }
  for (std::string &rtlPath : facts.rtlPaths) {
    auto path = namespacedBundlePath(rtlPath, *prefix);
    if (!path)
      return path.takeError();
    rtlPath = std::move(*path);
  }
  auto testbenchPath = namespacedBundlePath(mappedRtlTestbenchPath, *prefix);
  auto standaloneDriverPath =
      namespacedBundlePath(mappedRtlVerilatorDriverPath, *prefix);
  auto bridgedDriverPath =
      namespacedBundlePath(mappedRtlBridgedVerilatorDriverPath, *prefix);
  auto bridgeEngineSourcePath =
      namespacedBundlePath(mappedRtlBridgeEngineSourcePath, *prefix);
  auto simulatorExecutablePath =
      namespacedBundlePath(mappedRtlSimulatorExecutablePath, *prefix);
  auto resultPath = namespacedBundlePath(mappedRtlResultPath, *prefix);
  if (!testbenchPath || !standaloneDriverPath || !bridgedDriverPath ||
      !bridgeEngineSourcePath || !simulatorExecutablePath || !resultPath)
    return invalid("mapped RTL bundle path namespace is invalid");
  auto testbench = detail::renderMappedRtlTestbench(facts, *resultPath);
  if (!testbench)
    return testbench.takeError();
  auto standalone = detail::renderMappedRtlVerilatorDriver(
      facts, buildJobs, *testbenchPath, *simulatorExecutablePath);
  if (!standalone)
    return standalone.takeError();
  auto bridged = detail::renderMappedRtlBridgedVerilatorDriver(
      facts, buildJobs, *testbenchPath, *bridgeEngineSourcePath,
      *simulatorExecutablePath);
  if (!bridged)
    return bridged.takeError();
  return MappedRtlExecutionProjectionOrUnsupported{
      MappedRtlExecutionBundleProjection{
          std::move(facts.semanticInputs), std::move(*testbenchPath),
          std::move(*standaloneDriverPath), std::move(*bridgedDriverPath),
          std::move(*bridgeEngineSourcePath),
          std::move(*simulatorExecutablePath), std::move(*resultPath),
          std::move(*testbench), std::move(*standalone), std::move(*bridged)}};
}

llvm::Expected<external_tool::ExternalToolInvocationImportExpectation>
deriveMappedRtlExecutionImportExpectation(
    const MappedRtlExecutionClosure &closure, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix) {
  auto prefix = canonicalPathPrefix(pathPrefix);
  if (!prefix)
    return prefix.takeError();
  auto expectation =
      detail::deriveMappedRtlImportExpectation(closure, artifacts, blobs);
  if (!expectation)
    return expectation.takeError();
  for (external_tool::ExternalToolInvocationSemanticInput &input :
       expectation->semanticInputs) {
    auto path = namespacedBundlePath(input.relativePath, *prefix);
    if (!path)
      return path.takeError();
    input.relativePath = std::move(*path);
  }
  auto resultPath = namespacedBundlePath(mappedRtlResultPath, *prefix);
  if (!resultPath)
    return resultPath.takeError();
  expectation->declaredOutputs = {std::move(*resultPath)};
  return std::move(*expectation);
}

llvm::Expected<sim::SpatialEngineBoundaryResult>
projectMappedRtlSpatialEngineBoundaryResult(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlSimulationResult &result, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (result.terminal != MappedRtlTerminalStatus::Retired ||
      !result.retirementCycle || result.launchCycle > *result.retirementCycle ||
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
  auto retirement = evaluation::ExactRatio::get(*result.retirementCycle, 1);
  auto terminal = evaluation::ExactRatio::get(result.terminalCycle, 1);
  if (!launch)
    return launch.takeError();
  if (!retirement)
    return retirement.takeError();
  if (!terminal)
    return terminal.takeError();
  return sim::SpatialEngineBoundaryResult{
      sim::RetiredExecution{},
      std::move(*functional),
      sim::SpatialProgressObservations{
          sim::SpatialEventCoordinate{std::move(*launch), 0},
          sim::SpatialEventCoordinate{std::move(*retirement), 0},
          sim::SpatialEventCoordinate{std::move(*terminal), 0}},
      {}};
}

} // namespace loom::eda::open_source

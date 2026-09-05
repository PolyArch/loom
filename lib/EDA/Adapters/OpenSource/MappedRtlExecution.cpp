#include "MappedRtlSimulationInternal.h"

#include "Common/BlobDigest.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"
#include "Hardware/RTL/RtlModuleGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

namespace loom::eda::open_source {
namespace {

constexpr llvm::StringLiteral kCycleLimitOption = "max_cycles";
constexpr llvm::StringLiteral kBuildJobsOption = "build_jobs";
constexpr llvm::StringLiteral kBuildWorkersOption = "build_workers";
constexpr llvm::StringLiteral kModelThreadsOption = "model_threads";
constexpr std::uint64_t kDefaultCycleLimit = 1'000'000;
constexpr std::uint64_t kMaximumBuildWorkers = 4;
constexpr llvm::StringLiteral kVerilatorPreamblePath =
    "drivers/verilator-preamble.sv";
constexpr llvm::StringLiteral kVerilatorStateClassesPath =
    "drivers/verilator-state-classes.vlt";
constexpr llvm::StringLiteral kVerilatorStateClassesControl =
    "`verilator_config\nno_inline -module \"*\"\n";
constexpr llvm::StringLiteral kVerilationSelectionPolicy =
    "circt_instance_graph_flat_handshake_closure";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_execution_invalid: " + detail);
}

llvm::Expected<std::uint64_t>
positiveOption(const external_tool::LocalToolConfig &config,
               const external_tool::ExternalToolProviderDescriptor &provider,
               llvm::StringRef name, std::uint64_t defaultValue,
               std::uint64_t maximum) {
  const auto configured = config.tools.find(provider.binding.key);
  if (configured == config.tools.end())
    return defaultValue;
  const llvm::json::Value *value = configured->second.providerOptions.get(name);
  if (!value)
    return defaultValue;
  const std::optional<std::uint64_t> parsed = value->getAsUINT64();
  if (!parsed || *parsed == 0 || *parsed > maximum)
    return invalid(provider.binding.key + ".provider_options." + name +
                   " is outside its positive unsigned range");
  return *parsed;
}

llvm::Error invalidParallelism(const llvm::Twine &name) {
  return invalid(name + " must be 1, 2, 4, or 8");
}

/// A provider option from the closed parallelism domain.
llvm::Expected<std::uint64_t>
parallelismOption(const external_tool::LocalToolConfig &config,
                  const external_tool::ExternalToolProviderDescriptor &provider,
                  llvm::StringRef name, std::uint64_t defaultValue) {
  auto value = positiveOption(config, provider, name, defaultValue,
                              std::numeric_limits<std::uint64_t>::max());
  if (!value)
    return value.takeError();
  if (!isMappedRtlParallelismCount(*value))
    return invalidParallelism(provider.binding.key + ".provider_options." +
                              name);
  return *value;
}

/// The provider option names each simulator admits. Verilator owns the model
/// thread count and the gem5 build worker share; VCS compiles and simulates
/// without either; Xcelium elaborates and simulates single-threaded and
/// admits the cycle limit alone.
llvm::ArrayRef<llvm::StringLiteral>
admittedProviderOptions(MappedRtlHdlSimulator simulator) {
  static constexpr llvm::StringLiteral verilatorOptions[]{
      kCycleLimitOption, kBuildJobsOption, kBuildWorkersOption,
      kModelThreadsOption};
  static constexpr llvm::StringLiteral vcsOptions[]{kCycleLimitOption,
                                                    kBuildJobsOption};
  static constexpr llvm::StringLiteral xceliumOptions[]{kCycleLimitOption};
  switch (simulator) {
  case MappedRtlHdlSimulator::Verilator:
    return verilatorOptions;
  case MappedRtlHdlSimulator::Vcs:
    return vcsOptions;
  case MappedRtlHdlSimulator::Xcelium:
    return xceliumOptions;
  }
  llvm_unreachable("closed HDL simulator set");
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

/// One generated model contains the entire configured handshake network.
/// Split translation units remain independently compiled; they do not introduce
/// protect-lib scheduling boundaries or change the RTL dependency graph.
std::vector<std::string>
renderVerilationBuildCommand(const MappedRtlBuildTools &tools,
                             const MappedRtlSourcePlan &plan,
                             std::uint64_t buildJobs) {
  return {tools.make,
          "-C",
          plan.workDirectoryPath,
          "-f",
          plan.verilationMakefileName,
          "-j" + std::to_string(buildJobs),
          std::filesystem::path(mappedRtlSimulatorExecutablePath.str())
              .filename()
              .generic_string(),
          detail::mappedRtlParallelBuildVariable.str(),
          "CXX=" + tools.cxx,
          "LINK=" + tools.linkerInvocation,
          "AR=" + tools.archiver,
          "OBJCACHE="};
}

struct ParsedModule final {
  std::uint64_t bodyLines = 0;
  std::uint64_t transitiveBodyLines = 0;
  std::uint64_t rootMultiplicity = 0;
};

struct ParsedSource final {
  std::string originalPath;
  std::string derivedPath;
  std::string module;
  std::string bytes;
  BlobDigest originalDigest;
  std::uint64_t originalOffset = 0;
  std::vector<std::pair<std::string, std::uint64_t>> dependencies;
  std::uint64_t bodyLines = 0;
  std::uint64_t transitiveBodyLines = 0;
  std::uint64_t rootInstanceMultiplicity = 0;
};

BlobDigest digestSource(llvm::StringRef bytes) {
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
}

std::uint64_t saturatingAdd(std::uint64_t lhs, std::uint64_t rhs) {
  return lhs > std::numeric_limits<std::uint64_t>::max() - rhs
             ? std::numeric_limits<std::uint64_t>::max()
             : lhs + rhs;
}

std::uint64_t saturatingMultiply(std::uint64_t lhs, std::uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs
             ? std::numeric_limits<std::uint64_t>::max()
             : lhs * rhs;
}

std::uint64_t sourceLineCount(llvm::StringRef bytes) {
  if (bytes.empty())
    return 0;
  const std::uint64_t newlines =
      static_cast<std::uint64_t>(std::count(bytes.begin(), bytes.end(), '\n'));
  return newlines + (bytes.ends_with("\n") ? 0 : 1);
}

llvm::Expected<std::pair<std::vector<ParsedSource>, MappedRtlSourcePlan>>
deriveVerilationSourcePlan(detail::MappedRtlInvocationFacts &facts) {
  const hardware::rtl::RtlModuleGraphProjection &graph = facts.rtlModuleGraph;
  if (!graph.sourceDigest || graph.topModule >= graph.modules.size())
    return invalid("CIRCT RTL module graph is not bound to emitted source");
  const auto sourceInput =
      llvm::find_if(facts.semanticInputs, [&](const auto &file) {
        return llvm::is_contained(facts.rtlPaths, file.relativePath) &&
               file.contents.size() == graph.sourceByteCount &&
               digestSource(file.contents) == *graph.sourceDigest;
      });
  if (sourceInput == facts.semanticInputs.end())
    return invalid("no RTL payload matches the CIRCT module graph source");
  if (llvm::count_if(facts.semanticInputs, [&](const auto &file) {
        return llvm::is_contained(facts.rtlPaths, file.relativePath) &&
               file.contents.size() == graph.sourceByteCount &&
               digestSource(file.contents) == *graph.sourceDigest;
      }) != 1)
    return invalid("CIRCT module graph matches more than one RTL payload");
  const llvm::StringRef sourceBytes(sourceInput->contents);
  if (facts.top != graph.modules[graph.topModule].emittedName)
    return invalid("HardwareImplementation top disagrees with CIRCT graph");

  auto boundSource =
      hardware::rtl::bindRtlModuleGraphSource(graph, sourceBytes);
  if (!boundSource)
    return boundSource.takeError();
  const llvm::ArrayRef<llvm::StringRef> moduleBytes =
      boundSource->moduleBytes();
  for (const hardware::rtl::RtlModuleProjection &module : graph.modules)
    if (module.kind == hardware::rtl::RtlModuleDefinitionKind::External &&
        module.reachable)
      return invalid("reachable CIRCT module has no concrete definition");

  std::vector<ParsedModule> modules(graph.modules.size());
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal)
    if (graph.modules[ordinal].kind ==
        hardware::rtl::RtlModuleDefinitionKind::Concrete)
      modules[ordinal].bodyLines = sourceLineCount(moduleBytes[ordinal]);

  enum class VisitState : std::uint8_t { Unvisited, Active, Complete };
  std::vector<VisitState> visitStates(graph.modules.size(),
                                      VisitState::Unvisited);
  std::vector<std::set<std::size_t>> transitiveModules(graph.modules.size());
  std::vector<std::size_t> postorder;
  const auto visit = [&](auto &&self, std::size_t ordinal) -> llvm::Error {
    if (ordinal >= graph.modules.size())
      return invalid("CIRCT module dependency ordinal is out of range");
    if (visitStates[ordinal] == VisitState::Active)
      return invalid("CIRCT module dependency graph contains a cycle");
    if (visitStates[ordinal] == VisitState::Complete)
      return llvm::Error::success();
    visitStates[ordinal] = VisitState::Active;
    std::set<std::size_t> closure{ordinal};
    for (const hardware::rtl::RtlModuleDependency &dependency :
         graph.modules[ordinal].dependencies) {
      if (dependency.multiplicity == 0)
        return invalid("CIRCT module dependency has zero multiplicity");
      if (llvm::Error error = self(self, dependency.targetModule))
        return error;
      closure.insert(dependency.targetModule);
      closure.insert(transitiveModules[dependency.targetModule].begin(),
                     transitiveModules[dependency.targetModule].end());
    }
    std::uint64_t transitiveBodyLines = 0;
    for (std::size_t member : closure)
      transitiveBodyLines =
          saturatingAdd(transitiveBodyLines, modules[member].bodyLines);
    modules[ordinal].transitiveBodyLines = transitiveBodyLines;
    transitiveModules[ordinal] = std::move(closure);
    visitStates[ordinal] = VisitState::Complete;
    postorder.push_back(ordinal);
    return llvm::Error::success();
  };
  if (llvm::Error error = visit(visit, graph.topModule))
    return std::move(error);
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal)
    if (graph.modules[ordinal].reachable !=
        (visitStates[ordinal] == VisitState::Complete))
      return invalid("CIRCT module reachability cache is inconsistent");

  modules[graph.topModule].rootMultiplicity = 1;
  for (std::size_t ordinal : llvm::reverse(postorder)) {
    const std::uint64_t parentMultiplicity = modules[ordinal].rootMultiplicity;
    for (const hardware::rtl::RtlModuleDependency &dependency :
         graph.modules[ordinal].dependencies) {
      ParsedModule &target = modules[dependency.targetModule];
      target.rootMultiplicity = saturatingAdd(
          target.rootMultiplicity,
          saturatingMultiply(parentMultiplicity, dependency.multiplicity));
    }
  }

  MappedRtlSourcePlan plan;
  plan.sourcePath = sourceInput->relativePath;
  plan.sourceSha256 = formatBlobDigestHex(*graph.sourceDigest);
  plan.sourceByteCount = graph.sourceByteCount;
  plan.framingByteCount = graph.framingByteCount;
  plan.preamble = boundSource->preamble().str();
  // The complete root closure is one simulator model. A protect-lib block
  // makes every output depend on every input and would re-close the registered
  // handshake cuts owned by the Fabric lowering.
  const ParsedModule &root = modules[graph.topModule];
  plan.hardwareRootModule = graph.modules[graph.topModule].emittedName;
  plan.hardwareRootBodyLines = root.bodyLines;
  plan.hardwareRootTransitiveBodyLines = root.transitiveBodyLines;
  for (std::size_t member : transitiveModules[graph.topModule])
    plan.hardwareRootSourceClosureModules.push_back(
        graph.modules[member].emittedName);
  llvm::sort(plan.hardwareRootSourceClosureModules);

  plan.rtlLibraryDirectoryPath = "drivers/verilator-library";
  std::vector<ParsedSource> derivedSources;
  derivedSources.reserve(postorder.size());
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
    const hardware::rtl::RtlModuleProjection &definition =
        graph.modules[ordinal];
    if (!definition.reachable)
      continue;
    if (definition.emittedName.empty() ||
        llvm::StringRef(definition.emittedName).contains('/') ||
        llvm::StringRef(definition.emittedName).contains('\\') ||
        llvm::StringRef(definition.emittedName).contains('\0'))
      return invalid("RTL module name cannot identify a library source");
    std::string moduleSource = moduleBytes[ordinal].str();
    const std::string derivedPath =
        plan.rtlLibraryDirectoryPath + "/" + definition.emittedName + ".sv";
    std::vector<std::pair<std::string, std::uint64_t>> dependencies;
    dependencies.reserve(definition.dependencies.size());
    for (const hardware::rtl::RtlModuleDependency &dependency :
         definition.dependencies)
      dependencies.emplace_back(
          graph.modules[dependency.targetModule].emittedName,
          dependency.multiplicity);
    llvm::sort(dependencies);
    derivedSources.push_back(
        {sourceInput->relativePath, derivedPath, definition.emittedName,
         std::move(moduleSource), definition.emission->digest,
         definition.emission->offset, std::move(dependencies),
         modules[ordinal].bodyLines, modules[ordinal].transitiveBodyLines,
         modules[ordinal].rootMultiplicity});
  }
  plan.manifestPath = "drivers/verilator-hierarchy-plan.json";
  plan.verilatorControlPath = kVerilatorStateClassesPath.str();
  plan.workDirectoryPath = "work/verilator";
  plan.verilationMakefileName = "V" + mappedRtlHarnessTop.str() + ".mk";
  return std::pair{std::move(derivedSources), std::move(plan)};
}

std::string renderVerilationSourcePlan(const MappedRtlSourcePlan &plan,
                                       llvm::ArrayRef<ParsedSource> sources) {
  std::string text;
  llvm::raw_string_ostream output(text);
  llvm::json::OStream json(output, 2);
  const auto sourceFor = [&](llvm::StringRef module) -> const ParsedSource & {
    const auto found = llvm::find_if(sources, [&](const ParsedSource &source) {
      return source.module == module;
    });
    assert(found != sources.end() && "hierarchy closure module has no source");
    return *found;
  };
  const auto writeSourceIdentity = [&](const ParsedSource &source) {
    json.object([&] {
      json.attribute("module", source.module);
      json.attribute("source_offset", source.originalOffset);
      json.attribute("source_bytes", source.bytes.size());
      json.attribute("source_sha256",
                     formatBlobDigestHex(source.originalDigest));
    });
  };
  json.object([&] {
    json.attribute("schema", "loom.mapped_rtl_hierarchy_plan.3");
    json.attribute("selection_policy", kVerilationSelectionPolicy);
    json.attributeObject("rtl_source", [&] {
      json.attribute("path", plan.sourcePath);
      json.attribute("bytes", plan.sourceByteCount);
      json.attribute("sha256", plan.sourceSha256);
      json.attribute("preamble_bytes", plan.preamble.size());
      json.attribute("framing_bytes", plan.framingByteCount);
    });
    json.attribute("rtl_library_directory", plan.rtlLibraryDirectoryPath);
    json.attribute("verilator_control", plan.verilatorControlPath);
    json.attribute("work_directory", plan.workDirectoryPath);
    json.attribute("verilation_style", "flat");
    json.attribute("output_split_statements",
                   detail::mappedRtlOutputSplitStatements);
    json.attribute("output_group_count", detail::mappedRtlOutputGroupCount);
    json.attribute("verilation_makefile", plan.verilationMakefileName);
    json.attribute("verilator_top", mappedRtlHarnessTop);
    json.attributeObject("hardware_root", [&] {
      json.attribute("module", plan.hardwareRootModule);
      json.attribute("body_lines", plan.hardwareRootBodyLines);
      json.attribute("transitive_body_lines",
                     plan.hardwareRootTransitiveBodyLines);
      json.attributeArray("source_closure", [&] {
        for (const std::string &module : plan.hardwareRootSourceClosureModules)
          writeSourceIdentity(sourceFor(module));
      });
    });
    json.attributeArray("sources", [&] {
      for (const ParsedSource &source : sources)
        json.object([&] {
          json.attribute("original", source.originalPath);
          json.attribute("derived", source.derivedPath);
          json.attribute("module", source.module);
          json.attribute("body_lines", source.bodyLines);
          json.attribute("transitive_body_lines", source.transitiveBodyLines);
          json.attribute("root_instance_multiplicity",
                         source.rootInstanceMultiplicity);
          json.attribute("source_offset", source.originalOffset);
          json.attribute("source_bytes", source.bytes.size());
          json.attribute("source_sha256",
                         formatBlobDigestHex(source.originalDigest));
          json.attributeArray("direct_dependencies", [&] {
            for (const auto &dependency : source.dependencies)
              json.object([&] {
                json.attribute("module", dependency.first);
                json.attribute("multiplicity", dependency.second);
              });
          });
        });
    });
  });
  return output.str();
}

llvm::Expected<external_tool::ResolvedAuxiliaryToolExecutable>
resolveBuildTool(const external_tool::LocalToolConfig &localConfig,
                 llvm::StringRef slot, llvm::ArrayRef<llvm::StringRef> names) {
  const auto canonicalUsable =
      [](llvm::StringRef candidate) -> std::optional<std::string> {
    if (!llvm::sys::fs::can_execute(candidate))
      return std::nullopt;
    std::error_code error;
    const std::filesystem::path canonical =
        std::filesystem::canonical(candidate.str(), error);
    if (error)
      return std::nullopt;
    const std::string basename = canonical.filename().string();
    if (basename == "ccache" || basename == "sccache")
      return std::nullopt;
    return canonical.string();
  };
  std::optional<std::string> executable;
  std::string localKey = slot.str();
  const auto configured = localConfig.externalFiles.find(localKey);
  if (configured != localConfig.externalFiles.end()) {
    executable = configured->second;
  } else {
    for (llvm::StringRef name : names) {
      llvm::ErrorOr<std::string> candidate = llvm::sys::findProgramByName(name);
      if (candidate) {
        if (auto usable = canonicalUsable(*candidate)) {
          executable = std::move(*usable);
          break;
        }
      }
      // findProgramByName returns the first PATH hit.  A cache launcher can
      // hide the real compiler later in PATH, so continue the same search
      // after rejecting that launcher.
      auto environmentPath = llvm::sys::Process::GetEnv("PATH");
      if (!environmentPath)
        continue;
      llvm::SmallVector<llvm::StringRef, 32> pathEntries;
      llvm::StringRef(*environmentPath).split(pathEntries, ':', -1, false);
      for (llvm::StringRef directory : pathEntries) {
        llvm::SmallString<256> pathCandidate;
        if (directory.empty())
          pathCandidate = name;
        else {
          pathCandidate = directory;
          llvm::sys::path::append(pathCandidate, name);
        }
        if (auto usable = canonicalUsable(pathCandidate)) {
          executable = std::move(*usable);
          break;
        }
      }
      if (executable)
        break;
    }
  }
  if (!executable)
    return invalid("mapped RTL build tool is unavailable: " + slot);
  std::error_code error;
  const std::filesystem::path canonical =
      std::filesystem::canonical(*executable, error);
  if (error)
    return invalid("cannot canonicalize mapped RTL build tool: " + slot);
  const std::string basename = canonical.filename().string();
  if (basename == "ccache" || basename == "sccache")
    return invalid("mapped RTL build tool " + slot +
                   " must name a compiler or linker, not a cache launcher");
  auto fingerprint = external_tool::fingerprintExternalFile(canonical.string());
  if (!fingerprint)
    return fingerprint.takeError();
  return external_tool::ResolvedAuxiliaryToolExecutable{
      slot.str(), std::move(localKey), canonical.string(),
      std::move(*fingerprint)};
}

} // namespace

llvm::StringRef mappedRtlHdlSimulatorSpelling(MappedRtlHdlSimulator simulator) {
  switch (simulator) {
  case MappedRtlHdlSimulator::Verilator:
    return "verilator";
  case MappedRtlHdlSimulator::Vcs:
    return "vcs";
  case MappedRtlHdlSimulator::Xcelium:
    return "xcelium";
  }
  llvm_unreachable("closed HDL simulator set");
}

std::optional<MappedRtlHdlSimulator>
parseMappedRtlHdlSimulator(llvm::StringRef spelling) {
  for (MappedRtlHdlSimulator simulator : mappedRtlHdlSimulators)
    if (spelling == mappedRtlHdlSimulatorSpelling(simulator))
      return simulator;
  return std::nullopt;
}

const external_tool::ExternalToolProviderDescriptor &
mappedRtlHdlSimulatorProvider(MappedRtlHdlSimulator simulator) {
  switch (simulator) {
  case MappedRtlHdlSimulator::Verilator:
    return external_tool::verilatorProvider();
  case MappedRtlHdlSimulator::Vcs:
    return external_tool::vcsProvider();
  case MappedRtlHdlSimulator::Xcelium:
    return external_tool::xceliumProvider();
  }
  llvm_unreachable("closed HDL simulator set");
}

std::optional<MappedRtlHdlSimulator>
classifyMappedRtlHdlSimulator(llvm::StringRef stableHdlSimulatorBuildIdentity) {
  std::optional<MappedRtlHdlSimulator> classified;
  for (MappedRtlHdlSimulator simulator : mappedRtlHdlSimulators) {
    const auto &marker = mappedRtlHdlSimulatorProvider(simulator)
                             .versionProbe.requiredOutputSubstring;
    if (!marker || !stableHdlSimulatorBuildIdentity.contains(*marker))
      continue;
    if (classified)
      return std::nullopt;
    classified = simulator;
  }
  return classified;
}

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig,
    MappedRtlHdlSimulator simulator) {
  const auto &provider = mappedRtlHdlSimulatorProvider(simulator);
  const llvm::ArrayRef<llvm::StringLiteral> admitted =
      admittedProviderOptions(simulator);
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured != localConfig.tools.end()) {
    for (const auto &option : configured->second.providerOptions) {
      const llvm::StringRef name = option.first;
      if (!llvm::is_contained(admitted, name))
        return invalid(provider.binding.key +
                       ".provider_options contains unknown field " + name);
    }
  }
  auto cycleLimit = positiveOption(localConfig, provider, kCycleLimitOption,
                                   kDefaultCycleLimit,
                                   std::numeric_limits<std::uint64_t>::max());
  if (!cycleLimit)
    return cycleLimit.takeError();
  auto buildJobs = parallelismOption(localConfig, provider, kBuildJobsOption,
                                     mappedRtlDefaultBuildJobs);
  if (!buildJobs)
    return buildJobs.takeError();
  auto buildWorkers =
      positiveOption(localConfig, provider, kBuildWorkersOption,
                     mappedRtlDefaultBuildWorkers, kMaximumBuildWorkers);
  if (!buildWorkers)
    return buildWorkers.takeError();
  auto modelThreads = parallelismOption(
      localConfig, provider, kModelThreadsOption, mappedRtlDefaultModelThreads);
  if (!modelThreads)
    return modelThreads.takeError();
  return MappedRtlExecutionAttemptOptions{
      *cycleLimit, *buildJobs, *buildWorkers, *modelThreads,
      configured == localConfig.tools.end()
          ? std::vector<std::string>{}
          : configured->second.inheritEnvironment};
}

llvm::Expected<MappedRtlBuildTools>
resolveMappedRtlBuildTools(const external_tool::LocalToolConfig &localConfig) {
  auto make =
      resolveBuildTool(localConfig, "mapped_rtl_make", {"gmake", "make"});
  auto cxx = resolveBuildTool(localConfig, "mapped_rtl_cxx", {"clang++"});
  auto linker = resolveBuildTool(localConfig, "mapped_rtl_linker", {"clang++"});
  auto archiver =
      resolveBuildTool(localConfig, "mapped_rtl_archiver", {"llvm-ar", "ar"});
  if (!make || !cxx || !linker || !archiver)
    return llvm::joinErrors(
        llvm::joinErrors(make ? llvm::Error::success() : make.takeError(),
                         cxx ? llvm::Error::success() : cxx.takeError()),
        llvm::joinErrors(linker ? llvm::Error::success() : linker.takeError(),
                         archiver ? llvm::Error::success()
                                  : archiver.takeError()));
  MappedRtlBuildTools result{make->absolutePath,     cxx->absolutePath,
                             linker->absolutePath,   linker->absolutePath,
                             archiver->absolutePath, {}};
  if (llvm::StringRef(
          std::filesystem::path(linker->absolutePath).filename().string())
          .starts_with("clang"))
    result.linkerInvocation += " --driver-mode=g++";
  result.provenance.push_back(std::move(*make));
  result.provenance.push_back(std::move(*cxx));
  result.provenance.push_back(std::move(*linker));
  result.provenance.push_back(std::move(*archiver));
  llvm::sort(result.provenance, [](const auto &lhs, const auto &rhs) {
    return lhs.providerInputSlot < rhs.providerInputSlot;
  });
  return result;
}

llvm::Expected<MappedRtlExecutionProjectionOrUnsupported>
deriveMappedRtlExecutionBundleProjection(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlVerilationPlan &plan, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix) {
  if (plan.cycleLimit == 0)
    return invalid("execution cycle limit must be positive");
  if (!isMappedRtlParallelismCount(plan.buildJobs))
    return invalidParallelism("mapped RTL build jobs");
  if (!isMappedRtlParallelismCount(plan.modelThreads))
    return invalidParallelism("mapped RTL model threads");
  if (plan.verilatorExecutable.empty())
    return invalid("mapped RTL plan has no Verilator executable");
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
  facts.cycleLimit = plan.cycleLimit;
  auto sourcePlan = deriveVerilationSourcePlan(facts);
  if (!sourcePlan)
    return sourcePlan.takeError();
  auto prefix = canonicalPathPrefix(pathPrefix);
  if (!prefix)
    return prefix.takeError();
  for (external_tool::MaterializedBundleFile &file : facts.semanticInputs) {
    auto path = namespacedBundlePath(file.relativePath, *prefix);
    if (!path)
      return path.takeError();
    file.relativePath = std::move(*path);
  }
  for (ParsedSource &source : sourcePlan->first) {
    auto original = namespacedBundlePath(source.originalPath, *prefix);
    auto path = namespacedBundlePath(source.derivedPath, *prefix);
    if (!original || !path)
      return llvm::joinErrors(original ? llvm::Error::success()
                                       : original.takeError(),
                              path ? llvm::Error::success() : path.takeError());
    source.originalPath = std::move(*original);
    source.derivedPath = std::move(*path);
  }
  if (sourcePlan->first.empty())
    return invalid("CIRCT hierarchy has no reachable module source");
  sourcePlan->second.sourcePath = sourcePlan->first.front().originalPath;
  auto controlPath =
      namespacedBundlePath(sourcePlan->second.verilatorControlPath, *prefix);
  if (!controlPath)
    return controlPath.takeError();
  auto rtlLibraryDirectoryPath =
      namespacedBundlePath(sourcePlan->second.rtlLibraryDirectoryPath, *prefix);
  auto preamblePath = namespacedBundlePath(kVerilatorPreamblePath, *prefix);
  auto sourceManifestPath =
      namespacedBundlePath(sourcePlan->second.manifestPath, *prefix);
  auto workDirectoryPath =
      namespacedBundlePath(sourcePlan->second.workDirectoryPath, *prefix);
  if (!rtlLibraryDirectoryPath || !preamblePath || !sourceManifestPath ||
      !workDirectoryPath)
    return llvm::joinErrors(
        llvm::joinErrors(
            rtlLibraryDirectoryPath ? llvm::Error::success()
                                    : rtlLibraryDirectoryPath.takeError(),
            preamblePath ? llvm::Error::success() : preamblePath.takeError()),
        llvm::joinErrors(sourceManifestPath ? llvm::Error::success()
                                            : sourceManifestPath.takeError(),
                         workDirectoryPath ? llvm::Error::success()
                                           : workDirectoryPath.takeError()));
  sourcePlan->second.rtlLibraryDirectoryPath =
      std::move(*rtlLibraryDirectoryPath);
  sourcePlan->second.verilatorControlPath = std::move(*controlPath);
  sourcePlan->second.manifestPath = std::move(*sourceManifestPath);
  sourcePlan->second.workDirectoryPath = std::move(*workDirectoryPath);
  facts.rtlPaths = {*preamblePath};
  facts.rtlLibraryDirectories = {sourcePlan->second.rtlLibraryDirectoryPath};
  const std::string sourceManifest =
      renderVerilationSourcePlan(sourcePlan->second, sourcePlan->first);
  std::vector<external_tool::MaterializedBundleFile> toolLocalInputs;
  toolLocalInputs.reserve(sourcePlan->first.size() + 3);
  toolLocalInputs.push_back({sourcePlan->second.verilatorControlPath,
                            kVerilatorStateClassesControl.str(), std::nullopt,
                            false});
  toolLocalInputs.push_back(
      {*preamblePath, sourcePlan->second.preamble, std::nullopt, false});
  for (ParsedSource &source : sourcePlan->first) {
    toolLocalInputs.push_back(
        {source.derivedPath, std::move(source.bytes), std::nullopt, false});
  }
  toolLocalInputs.push_back(
      {sourcePlan->second.manifestPath, sourceManifest, std::nullopt, false});
  // The rendered program images are semantic inputs materialized with the
  // facts; the testbench addresses them through their namespaced paths.
  std::vector<std::string> configurationProgramPaths;
  configurationProgramPaths.reserve(facts.configurationProgramPaths.size());
  for (const std::string &unnamespacedPath : facts.configurationProgramPaths) {
    auto path = namespacedBundlePath(unnamespacedPath, *prefix);
    if (!path)
      return path.takeError();
    configurationProgramPaths.push_back(std::move(*path));
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
  auto transportReceiptPath = namespacedBundlePath(
      ::loom::eda::open_source::mappedRtlConfigurationTransportReceiptPath,
      *prefix);
  if (!testbenchPath || !standaloneDriverPath || !bridgedDriverPath ||
      !bridgeEngineSourcePath || !simulatorExecutablePath || !resultPath)
    return invalid("mapped RTL bundle path namespace is invalid");
  if (!transportReceiptPath)
    return transportReceiptPath.takeError();
  auto testbench = detail::renderMappedRtlTestbench(
      facts, configurationProgramPaths, *resultPath, *transportReceiptPath);
  if (!testbench)
    return testbench.takeError();
  auto standalone = detail::renderMappedRtlVerilatorDriver(
      facts, plan, sourcePlan->second.verilatorControlPath, *testbenchPath,
      *simulatorExecutablePath, std::nullopt);
  if (!standalone)
    return standalone.takeError();
  auto bridged = detail::renderMappedRtlVerilatorDriver(
      facts, plan, sourcePlan->second.verilatorControlPath, *testbenchPath,
      *simulatorExecutablePath,
      llvm::StringRef(*bridgeEngineSourcePath));
  if (!bridged)
    return bridged.takeError();
  MappedRtlExecutionBundleProjection result;
  result.buildCommand = renderVerilationBuildCommand(
      plan.buildTools, sourcePlan->second, plan.buildJobs);
  result.semanticInputs = std::move(facts.semanticInputs);
  result.toolLocalInputs = std::move(toolLocalInputs);
  result.sourcePlan = std::move(sourcePlan->second);
  result.configurationProgramPaths = std::move(configurationProgramPaths);
  result.testbenchPath = std::move(*testbenchPath);
  result.standaloneVerilatorDriverPath = std::move(*standaloneDriverPath);
  result.bridgedVerilatorDriverPath = std::move(*bridgedDriverPath);
  result.bridgeEngineSourcePath = std::move(*bridgeEngineSourcePath);
  result.simulatorExecutablePath = std::move(*simulatorExecutablePath);
  result.resultPath = std::move(*resultPath);
  result.configurationTransportReceiptPath = std::move(*transportReceiptPath);
  result.testbench = std::move(*testbench);
  result.standaloneVerilatorDriver = std::move(*standalone);
  result.bridgedVerilatorDriver = std::move(*bridged);
  return MappedRtlExecutionProjectionOrUnsupported{std::move(result)};
}

namespace {

/// The shared derivation of an event-driven member's bundle: the invocation
/// facts, the harness, and the member's argument file rendered from those
/// facts. The caller adds the member's command schedule.
llvm::Expected<MappedRtlEventDrivenProjectionOrUnsupported>
deriveEventDrivenBundleProjection(
    const MappedRtlExecutionClosure &closure, std::uint64_t cycleLimit,
    llvm::StringRef driverPath, const ArtifactStore &artifacts,
    const BlobStore &blobs,
    llvm::function_ref<
        llvm::Expected<std::string>(const detail::MappedRtlInvocationFacts &)>
        renderDriver) {
  if (cycleLimit == 0)
    return invalid("execution cycle limit must be positive");
  auto factsOrUnsupported =
      detail::deriveMappedRtlInvocationFacts(closure, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<evaluation::UnsupportedEvidence>(&*factsOrUnsupported))
    return MappedRtlEventDrivenProjectionOrUnsupported{*unsupported};
  detail::MappedRtlInvocationFacts facts =
      std::get<detail::MappedRtlInvocationFacts>(
          std::move(*factsOrUnsupported));
  facts.cycleLimit = cycleLimit;
  auto testbench = detail::renderMappedRtlTestbench(
      facts, facts.configurationProgramPaths, mappedRtlResultPath,
      ::loom::eda::open_source::mappedRtlConfigurationTransportReceiptPath);
  if (!testbench)
    return testbench.takeError();
  auto driver = renderDriver(facts);
  if (!driver)
    return driver.takeError();
  MappedRtlEventDrivenBundleProjection result;
  result.semanticInputs = std::move(facts.semanticInputs);
  result.testbenchPath = mappedRtlTestbenchPath.str();
  result.driverPath = driverPath.str();
  result.resultPath = mappedRtlResultPath.str();
  result.configurationTransportReceiptPath =
      ::loom::eda::open_source::mappedRtlConfigurationTransportReceiptPath
          .str();
  result.testbench = std::move(*testbench);
  result.driver = std::move(*driver);
  return MappedRtlEventDrivenProjectionOrUnsupported{std::move(result)};
}

} // namespace

llvm::Expected<MappedRtlEventDrivenProjectionOrUnsupported>
deriveMappedRtlVcsBundleProjection(const MappedRtlExecutionClosure &closure,
                                   const MappedRtlVcsCompilationPlan &plan,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs) {
  if (!isMappedRtlParallelismCount(plan.buildJobs))
    return invalidParallelism("mapped RTL build jobs");
  if (plan.vcsExecutable.empty())
    return invalid("mapped RTL plan has no VCS executable");
  auto projection = deriveEventDrivenBundleProjection(
      closure, plan.cycleLimit, mappedRtlVcsDriverPath, artifacts, blobs,
      [&](const detail::MappedRtlInvocationFacts &facts) {
        return detail::renderMappedRtlVcsDriver(
            facts, plan, mappedRtlTestbenchPath, mappedRtlVcsWorkDirectoryPath,
            mappedRtlVcsSimulatorExecutablePath);
      });
  if (!projection)
    return projection.takeError();
  auto *result =
      std::get_if<MappedRtlEventDrivenBundleProjection>(&*projection);
  if (!result)
    return projection;
  result->toolProducedExecutables = {mappedRtlVcsSimulatorExecutablePath.str()};
  // The mandatory 64-bit architecture token is projected into the structured
  // command by the provider descriptor's contract, never left to the
  // argument file. `-file` admits every option in the file; `-f` admits only
  // sources and a subset of options.
  result->compileCommand = {plan.vcsExecutable, "-full64", "-file",
                            result->driverPath};
  // The simulator records interactive commands in a key file in its working
  // directory unless told not to, and re-executes itself to disable address
  // space randomization for save/restore unless that feature is off; the
  // bundle root admits no undeclared product and the harness saves nothing.
  result->simulationCommand = {mappedRtlVcsSimulatorExecutablePath.str(), "-k",
                               "off", "-no_save"};
  return projection;
}

llvm::Expected<MappedRtlEventDrivenProjectionOrUnsupported>
deriveMappedRtlXceliumBundleProjection(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlXceliumElaborationPlan &plan, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (plan.xrunExecutable.empty())
    return invalid("mapped RTL plan has no Xcelium executable");
  auto projection = deriveEventDrivenBundleProjection(
      closure, plan.cycleLimit, mappedRtlXceliumDriverPath, artifacts, blobs,
      [&](const detail::MappedRtlInvocationFacts &facts) {
        return detail::renderMappedRtlXceliumDriver(
            facts, mappedRtlTestbenchPath,
            mappedRtlXceliumLibraryDirectoryPath);
      });
  if (!projection)
    return projection.takeError();
  auto *result =
      std::get_if<MappedRtlEventDrivenBundleProjection>(&*projection);
  if (!result)
    return projection;
  // The mandatory 64-bit token is a command token like VCS's `-full64`. The
  // elaboration command parses and elaborates the argument file's sources
  // into a snapshot inside the library directory and does not simulate; the
  // simulation command runs the last elaborated snapshot of that library
  // through the same launcher, so the bundle lists no tool-produced
  // executable. The launcher writes a log, a key file, and a history file
  // into its working directory unless each is turned off; the bundle root
  // admits no undeclared product, and stdout is already captured.
  result->compileCommand = {plan.xrunExecutable, "-64bit", "-elaborate", "-f",
                            result->driverPath};
  result->simulationCommand = {plan.xrunExecutable,
                               "-64bit",
                               "-R",
                               "-xmlibdirname",
                               mappedRtlXceliumLibraryDirectoryPath.str(),
                               "-nolog",
                               "-nokey",
                               "-nohistory",
                               "-noenvhistory"};
  return projection;
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
  auto receiptPath =
      namespacedBundlePath(mappedRtlConfigurationTransportReceiptPath, *prefix);
  if (!receiptPath)
    return receiptPath.takeError();
  expectation->declaredOutputs.push_back(std::move(*receiptPath));
  return std::move(*expectation);
}

llvm::Error validateMappedRtlConfigurationTransportReceipt(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlConfigurationTransportReceipt &receipt,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto factsOrUnsupported =
      detail::deriveMappedRtlInvocationFacts(closure, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<evaluation::UnsupportedEvidence>(&*factsOrUnsupported))
    return invalid("configuration transport receipt cannot be validated for "
                   "an unsupported mapped RTL closure: " +
                   evaluation::toString(unsupported->reason));
  const auto &facts =
      std::get<detail::MappedRtlInvocationFacts>(*factsOrUnsupported);
  if (receipt.programs.size() != facts.configurationPrograms.size())
    return invalid("configuration transport receipt program count disagrees "
                   "with the ConfigurationABI");
  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms)) {
    const auto &observed = receipt.programs[ordinal];
    const std::uint64_t expectedWords = program.layout.payloadWordCount;
    if (observed.payloadWrites != expectedWords)
      return invalid("configuration program " + std::to_string(ordinal) +
                     " payload write count disagrees with the "
                     "ConfigurationABI program payloadWordCount");
    if (observed.atomicCommits != 1)
      return invalid("configuration program " + std::to_string(ordinal) +
                     " has an invalid atomic commit count");
    if (observed.activeWordComparisons != expectedWords)
      return invalid("configuration program " + std::to_string(ordinal) +
                     " active-word comparison count disagrees with the "
                     "ConfigurationABI program payloadWordCount");
    if (observed.passingStatusReads != 1)
      return invalid("configuration program " + std::to_string(ordinal) +
                     " has an invalid passing status read count");
  }
  return llvm::Error::success();
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

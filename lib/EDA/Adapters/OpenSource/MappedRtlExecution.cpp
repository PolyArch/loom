#include "MappedRtlSimulationInternal.h"

#include "Common/BlobDigest.h"
#include "EDA/Adapters/OpenSource/MappedRtlHierarchyLauncher.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::eda::open_source {
namespace {

constexpr llvm::StringLiteral kCycleLimitOption = "max_cycles";
constexpr llvm::StringLiteral kBuildJobsOption = "build_jobs";
constexpr llvm::StringLiteral kBuildWorkersOption = "build_workers";
constexpr llvm::StringLiteral kModelThreadsOption = "model_threads";
constexpr std::uint64_t kDefaultCycleLimit = 1'000'000;
constexpr std::uint64_t kMaximumParallelism = 8;
constexpr std::uint64_t kMaximumBuildWorkers = 4;
constexpr std::uint64_t kHierarchyBodyLineThreshold = 10'000;
constexpr std::uint64_t kHierarchyMinimumReuseMultiplicity = 8;
constexpr std::uint64_t kHierarchyMultiplicityWeightThreshold = 25'000;
constexpr std::uint64_t kHierarchyRootClosureBodyLineBudget = 100'000;
constexpr std::uint64_t kHierarchyRootClosureByteBudget = 4'000'000;
constexpr std::size_t kHierarchyMaximumBlockCount = 128;
constexpr llvm::StringLiteral kHierarchySelectionPolicy =
    "circt_instance_graph_root_closure_rebalanced";

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

/// The closed parallelism domain shared by Verilation jobs, make jobs, and
/// simulation model threads.
bool isParallelismCount(std::uint64_t value) {
  return value == 1 || value == 2 || value == 4 || value == 8;
}

llvm::Error invalidParallelism(const llvm::Twine &name) {
  return invalid(name + " must be 1, 2, 4, or 8");
}

/// Verilator joins make flags into one unquoted make command line, so a
/// variable value must be a single shell word.
llvm::Error validateMakeVariableValue(llvm::StringRef name,
                                      llvm::StringRef value) {
  if (value.empty() || value.contains('\0') || value.contains('\n') ||
      value.contains(' ') || value.contains('\t') || value.contains('\'') ||
      value.contains('"') || value.contains('\\'))
    return invalid("make variable " + name +
                   " is not representable as one Verilator make flag");
  return llvm::Error::success();
}

/// The make command-line variables that bind the generated hierarchy makefile
/// to the frozen hierarchy launcher, the frozen Verilator executable, and the
/// harness path the launcher removes from child argument files. GNU make
/// exports command-line variables to every recipe, which is how the launcher
/// receives its configuration.
llvm::Expected<std::vector<std::string>>
renderHierarchyLauncherMakeVariables(const MappedRtlBuildTools &tools,
                                     llvm::StringRef verilatorExecutable,
                                     llvm::StringRef testbenchPath) {
  const std::array<std::pair<llvm::StringRef, llvm::StringRef>, 3> bindings{{
      {verilatorHierarchyLauncherVariable, tools.hierarchyLauncher},
      {mappedRtlHierarchyVerilatorVariable, verilatorExecutable},
      {mappedRtlHierarchyTestbenchVariable, testbenchPath},
  }};
  std::vector<std::string> variables;
  variables.reserve(bindings.size());
  for (const auto &[name, value] : bindings) {
    if (llvm::Error error = validateMakeVariableValue(name, value))
      return std::move(error);
    variables.push_back((name + "=" + value).str());
  }
  return variables;
}

/// The target of the generated makefile: hierarchical Verilation owns the
/// child blocks and the root through `hier_build`; flat Verilation has one
/// makefile whose target is the simulator executable.
std::string verilationBuildTarget(MappedRtlVerilationStyle style) {
  if (style == MappedRtlVerilationStyle::Hierarchical)
    return "hier_build";
  return std::filesystem::path(mappedRtlSimulatorExecutablePath.str())
      .filename()
      .generic_string();
}

std::vector<std::string>
renderVerilationBuildCommand(const MappedRtlBuildTools &tools,
                             const MappedRtlHierarchyPlan &hierarchy,
                             std::uint64_t buildJobs,
                             llvm::ArrayRef<std::string> launcherVariables) {
  std::vector<std::string> command{
      tools.make,
      "-C",
      hierarchy.workDirectoryPath,
      "-f",
      hierarchy.verilationMakefileName,
      "-j" + std::to_string(buildJobs),
      verilationBuildTarget(hierarchy.style),
      "CXX=" + tools.cxx,
      "LINK=" + tools.linkerInvocation,
      "AR=" + tools.archiver,
      "OBJCACHE="};
  command.insert(command.end(), launcherVariables.begin(),
                 launcherVariables.end());
  return command;
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
  std::string originalBytes;
  std::string derivedBytes;
  BlobDigest originalDigest;
  std::uint64_t originalOffset = 0;
  std::vector<std::pair<std::string, std::uint64_t>> dependencies;
  bool hierarchyBlock = false;
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

llvm::Expected<std::string> annotateHierarchicalBlock(llvm::StringRef bytes,
                                                      llvm::StringRef module) {
  std::size_t definitionOffset = 0;
  while (definitionOffset < bytes.size()) {
    while (definitionOffset < bytes.size() &&
           std::isspace(static_cast<unsigned char>(bytes[definitionOffset])))
      ++definitionOffset;
    if (bytes.substr(definitionOffset).starts_with("//")) {
      const std::size_t newline = bytes.find('\n', definitionOffset + 2);
      definitionOffset =
          newline == llvm::StringRef::npos ? bytes.size() : newline + 1;
      continue;
    }
    if (bytes.substr(definitionOffset).starts_with("/*")) {
      const std::size_t end = bytes.find("*/", definitionOffset + 2);
      if (end == llvm::StringRef::npos)
        return invalid("framed RTL module has an unterminated prelude comment");
      definitionOffset = end + 2;
      continue;
    }
    break;
  }
  llvm::StringRef header = bytes.drop_front(definitionOffset);
  if (!header.consume_front("module"))
    return invalid("framed RTL module does not start with its definition");
  if (header.empty() ||
      !std::isspace(static_cast<unsigned char>(header.front())))
    return invalid("framed RTL module has a malformed definition header");
  header = header.ltrim();
  const std::size_t nameEnd = header.find_first_of(" #(\t\r\n");
  if (header.take_front(nameEnd) != module)
    return invalid("framed RTL module name disagrees with the CIRCT graph");

  enum class LexState : std::uint8_t {
    Normal,
    LineComment,
    BlockComment,
    String,
  };
  LexState state = LexState::Normal;
  bool escaped = false;
  std::optional<std::size_t> terminator;
  for (std::size_t index = 0; index != bytes.size(); ++index) {
    const char character = bytes[index];
    const char next = index + 1 < bytes.size() ? bytes[index + 1] : '\0';
    switch (state) {
    case LexState::Normal:
      if (character == '/' && next == '/') {
        state = LexState::LineComment;
        ++index;
      } else if (character == '/' && next == '*') {
        state = LexState::BlockComment;
        ++index;
      } else if (character == '"') {
        state = LexState::String;
        escaped = false;
      } else if (character == ';') {
        terminator = index;
      }
      break;
    case LexState::LineComment:
      if (character == '\n')
        state = LexState::Normal;
      break;
    case LexState::BlockComment:
      if (character == '*' && next == '/') {
        state = LexState::Normal;
        ++index;
      }
      break;
    case LexState::String:
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        state = LexState::Normal;
      }
      break;
    }
    if (terminator)
      break;
  }
  if (!terminator || state != LexState::Normal)
    return invalid("framed RTL module header has no exact terminator");
  std::string result;
  result.reserve(bytes.size() + 32);
  result.append(bytes.data(), *terminator + 1);
  result += "\n  /*verilator hier_block*/";
  result.append(bytes.data() + *terminator + 1, bytes.size() - *terminator - 1);
  return result;
}

llvm::Expected<std::pair<std::vector<ParsedSource>, MappedRtlHierarchyPlan>>
deriveHierarchyPlan(detail::MappedRtlInvocationFacts &facts, llvm::StringRef) {
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

  const auto verifyRange =
      [&](const hardware::rtl::RtlModuleEmissionRange &range,
          llvm::StringRef owner) -> llvm::Expected<llvm::StringRef> {
    if (range.offset > sourceBytes.size() ||
        range.byteCount > sourceBytes.size() - range.offset)
      return invalid(owner + " source range is outside the RTL payload");
    llvm::StringRef bytes =
        sourceBytes.substr(static_cast<std::size_t>(range.offset),
                           static_cast<std::size_t>(range.byteCount));
    if (digestSource(bytes) != range.digest)
      return invalid(owner + " source range digest is inconsistent");
    return bytes;
  };
  std::uint64_t accounted = graph.framingByteCount;
  if (graph.preamble) {
    if (graph.preamble->offset != 0)
      return invalid("CIRCT RTL preamble does not start at byte zero");
    auto bytes = verifyRange(*graph.preamble, "RTL preamble");
    if (!bytes)
      return bytes.takeError();
    accounted = saturatingAdd(accounted, graph.preamble->byteCount);
  }
  std::vector<llvm::StringRef> moduleBytes(graph.modules.size());
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
    const hardware::rtl::RtlModuleProjection &module = graph.modules[ordinal];
    if (module.kind == hardware::rtl::RtlModuleDefinitionKind::External) {
      if (module.emission)
        return invalid("external CIRCT module unexpectedly owns source bytes");
      if (module.reachable)
        return invalid("reachable CIRCT module has no concrete definition");
      continue;
    }
    if (!module.emission)
      return invalid("concrete CIRCT module has no emitted source range");
    auto bytes = verifyRange(*module.emission, module.emittedName);
    if (!bytes)
      return bytes.takeError();
    moduleBytes[ordinal] = *bytes;
    accounted = saturatingAdd(accounted, module.emission->byteCount);
    ranges.emplace_back(
        module.emission->offset,
        saturatingAdd(module.emission->offset, module.emission->byteCount));
  }
  if (accounted != graph.sourceByteCount ||
      graph.sourceByteCount != sourceBytes.size())
    return invalid("CIRCT module graph does not cover the exact RTL payload");
  llvm::sort(ranges);
  for (std::size_t index = 1; index != ranges.size(); ++index)
    if (ranges[index - 1].second > ranges[index].first)
      return invalid("CIRCT module source ranges overlap");

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

  MappedRtlHierarchyPlan plan;
  plan.selectionPolicy = kHierarchySelectionPolicy.str();
  plan.sourcePath = sourceInput->relativePath;
  plan.sourceSha256 = formatBlobDigestHex(*graph.sourceDigest);
  plan.sourceByteCount = graph.sourceByteCount;
  plan.framingByteCount = graph.framingByteCount;
  plan.preambleByteCount = graph.preamble ? graph.preamble->byteCount : 0;
  std::set<std::size_t> selected;
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
    const ParsedModule &module = modules[ordinal];
    if (!graph.modules[ordinal].reachable || ordinal == graph.topModule)
      continue;
    const std::uint64_t weight =
        saturatingMultiply(module.transitiveBodyLines, module.rootMultiplicity);
    if (module.bodyLines < kHierarchyBodyLineThreshold &&
        (module.rootMultiplicity < kHierarchyMinimumReuseMultiplicity ||
         weight < kHierarchyMultiplicityWeightThreshold))
      continue;
    selected.insert(ordinal);
  }
  const auto sourceClosureFor = [&](std::size_t start,
                                    const std::set<std::size_t> &blocks) {
    std::set<std::size_t> closure{start};
    std::vector<std::size_t> work{start};
    for (std::size_t index = 0; index != work.size(); ++index) {
      const std::size_t ordinal = work[index];
      for (const hardware::rtl::RtlModuleDependency &dependency :
           graph.modules[ordinal].dependencies) {
        if (dependency.targetModule != start &&
            blocks.count(dependency.targetModule) != 0)
          continue;
        if (!closure.insert(dependency.targetModule).second)
          continue;
        work.push_back(dependency.targetModule);
      }
    }
    return closure;
  };
  const auto closureBodyLines = [&](const std::set<std::size_t> &closure) {
    std::uint64_t result = 0;
    for (std::size_t ordinal : closure)
      result = saturatingAdd(result, modules[ordinal].bodyLines);
    return result;
  };
  const auto closureBytes = [&](const std::set<std::size_t> &closure) {
    std::uint64_t result = 0;
    for (std::size_t ordinal : closure)
      result = saturatingAdd(result, moduleBytes[ordinal].size());
    return result;
  };
  const std::set<std::size_t> baselineSelected = selected;
  const std::set<std::size_t> baselineRootClosure =
      sourceClosureFor(graph.topModule, baselineSelected);
  plan.baselineBlockCount = baselineSelected.size();
  plan.baselineRootSourceClosureModuleCount = baselineRootClosure.size();
  plan.baselineRootSourceClosureBodyLines =
      closureBodyLines(baselineRootClosure);
  plan.baselineRootSourceClosureBytes = closureBytes(baselineRootClosure);

  // Rebalance only along the exact CIRCT dependency graph. A candidate is
  // admitted when it removes at least one source body line from the root
  // closure; the score prefers the greatest reduction per child transitive
  // body, with stable graph-name tie breaking.
  std::set<std::size_t> rootSourceClosure =
      sourceClosureFor(graph.topModule, selected);
  while ((closureBodyLines(rootSourceClosure) >
              kHierarchyRootClosureBodyLineBudget ||
          closureBytes(rootSourceClosure) > kHierarchyRootClosureByteBudget) &&
         selected.size() < kHierarchyMaximumBlockCount) {
    const std::uint64_t currentBodyLines = closureBodyLines(rootSourceClosure);
    std::optional<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                             std::uint64_t, std::string>> bestScore;
    std::optional<std::size_t> bestOrdinal;
    for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
      if (ordinal == graph.topModule || !graph.modules[ordinal].reachable ||
          selected.count(ordinal) != 0 ||
          rootSourceClosure.count(ordinal) == 0)
        continue;
      std::set<std::size_t> trialBlocks = selected;
      trialBlocks.insert(ordinal);
      const std::set<std::size_t> trialClosure =
          sourceClosureFor(graph.topModule, trialBlocks);
      const std::uint64_t trialBodyLines = closureBodyLines(trialClosure);
      if (trialBodyLines >= currentBodyLines)
        continue;
      const std::uint64_t benefit = currentBodyLines - trialBodyLines;
      const ParsedModule &candidate = modules[ordinal];
      const std::uint64_t denominator =
          saturatingAdd(candidate.transitiveBodyLines, 1);
      const std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::string>
          score{benefit / denominator, benefit, candidate.rootMultiplicity,
                candidate.bodyLines, graph.modules[ordinal].emittedName};
      if (!bestScore || score > *bestScore) {
        bestScore = score;
        bestOrdinal = ordinal;
      }
    }
    if (!bestOrdinal)
      break;
    selected.insert(*bestOrdinal);
    rootSourceClosure = sourceClosureFor(graph.topModule, selected);
  }
  plan.rootSourceClosureBodyLines = closureBodyLines(rootSourceClosure);
  plan.rootSourceClosureBytes = closureBytes(rootSourceClosure);
  const ParsedModule &root = modules[graph.topModule];
  plan.hardwareRootModule = graph.modules[graph.topModule].emittedName;
  plan.hardwareRootBodyLines = root.bodyLines;
  plan.hardwareRootTransitiveBodyLines = root.transitiveBodyLines;
  for (std::size_t member : rootSourceClosure)
    plan.hardwareRootSourceClosureModules.push_back(
        graph.modules[member].emittedName);
  llvm::sort(plan.hardwareRootSourceClosureModules);
  for (std::size_t selectedModule : selected) {
    const ParsedModule &module = modules[selectedModule];
    std::set<std::size_t> sourceClosure{selectedModule};
    std::vector<std::size_t> closureWork{selectedModule};
    for (std::size_t ordinal = 0; ordinal != closureWork.size(); ++ordinal) {
      for (const hardware::rtl::RtlModuleDependency &dependency :
           graph.modules[closureWork[ordinal]].dependencies) {
        if (selected.count(dependency.targetModule) != 0 ||
            !sourceClosure.insert(dependency.targetModule).second)
          continue;
        closureWork.push_back(dependency.targetModule);
      }
    }
    std::vector<std::string> names;
    names.reserve(sourceClosure.size());
    for (std::size_t member : sourceClosure)
      names.push_back(graph.modules[member].emittedName);
    llvm::sort(names);
    plan.blocks.push_back({graph.modules[selectedModule].emittedName,
                           module.bodyLines, module.transitiveBodyLines,
                           module.rootMultiplicity, std::move(names)});
  }
  llvm::sort(plan.blocks, [](const auto &lhs, const auto &rhs) {
    return lhs.module < rhs.module;
  });

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
    std::string derived = moduleSource;
    if (selected.count(ordinal) != 0) {
      auto annotated =
          annotateHierarchicalBlock(moduleSource, definition.emittedName);
      if (!annotated)
        return annotated.takeError();
      derived = std::move(*annotated);
    }
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
         std::move(moduleSource), std::move(derived),
         definition.emission->digest, definition.emission->offset,
         std::move(dependencies),
         selected.count(ordinal) != 0, modules[ordinal].bodyLines,
         modules[ordinal].transitiveBodyLines,
         modules[ordinal].rootMultiplicity});
    plan.derivedRtlPaths.push_back(derivedPath);
  }
  plan.manifestPath = "drivers/verilator-hierarchy-plan.json";
  plan.workDirectoryPath = "work/verilator";
  plan.style = plan.blocks.empty() ? MappedRtlVerilationStyle::Flat
                                   : MappedRtlVerilationStyle::Hierarchical;
  plan.verilationMakefileName =
      "V" + mappedRtlHarnessTop.str() +
      (plan.style == MappedRtlVerilationStyle::Hierarchical ? "_hier.mk"
                                                            : ".mk");
  return std::pair{std::move(derivedSources), std::move(plan)};
}

std::string renderHierarchyPlan(const MappedRtlHierarchyPlan &plan,
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
      json.attribute("source_bytes", source.originalBytes.size());
      json.attribute("source_sha256",
                     formatBlobDigestHex(source.originalDigest));
    });
  };
  json.object([&] {
    json.attribute("schema", "loom.mapped_rtl_hierarchy_plan.2");
    json.attribute("selection_policy", plan.selectionPolicy);
    json.attribute("body_line_threshold", kHierarchyBodyLineThreshold);
    json.attribute("minimum_reuse_multiplicity",
                   kHierarchyMinimumReuseMultiplicity);
    json.attribute("multiplicity_weight_threshold",
                   kHierarchyMultiplicityWeightThreshold);
    json.attributeObject("root_closure_rebalance", [&] {
      json.attribute("body_line_budget", kHierarchyRootClosureBodyLineBudget);
      json.attribute("byte_budget", kHierarchyRootClosureByteBudget);
      json.attribute("maximum_block_count", kHierarchyMaximumBlockCount);
      json.attributeObject("baseline", [&] {
        json.attribute("block_count", plan.baselineBlockCount);
        json.attribute("source_closure_modules",
                       plan.baselineRootSourceClosureModuleCount);
        json.attribute("source_closure_body_lines",
                       plan.baselineRootSourceClosureBodyLines);
        json.attribute("source_closure_bytes",
                       plan.baselineRootSourceClosureBytes);
      });
      json.attributeObject("selected", [&] {
        json.attribute("block_count", plan.blocks.size());
        json.attribute("source_closure_modules",
                       plan.hardwareRootSourceClosureModules.size());
        json.attribute("source_closure_body_lines",
                       plan.rootSourceClosureBodyLines);
        json.attribute("source_closure_bytes", plan.rootSourceClosureBytes);
      });
    });
    json.attributeObject("rtl_source", [&] {
      json.attribute("path", plan.sourcePath);
      json.attribute("bytes", plan.sourceByteCount);
      json.attribute("sha256", plan.sourceSha256);
      json.attribute("preamble_bytes", plan.preambleByteCount);
      json.attribute("framing_bytes", plan.framingByteCount);
    });
    json.attribute("rtl_library_directory", plan.rtlLibraryDirectoryPath);
    json.attribute("work_directory", plan.workDirectoryPath);
    json.attribute("verilation_style",
                   plan.style == MappedRtlVerilationStyle::Hierarchical
                       ? "hierarchical"
                       : "flat");
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
          json.attribute("hierarchy_block", source.hierarchyBlock);
          json.attribute("body_lines", source.bodyLines);
          json.attribute("transitive_body_lines", source.transitiveBodyLines);
          json.attribute("root_instance_multiplicity",
                         source.rootInstanceMultiplicity);
          json.attribute("source_offset", source.originalOffset);
          json.attribute("source_bytes", source.originalBytes.size());
          json.attribute("source_sha256",
                         formatBlobDigestHex(source.originalDigest));
          json.attribute("derived_bytes", source.derivedBytes.size());
          json.attribute("derived_sha256", formatBlobDigestHex(digestSource(
                                               source.derivedBytes)));
          json.attributeArray("direct_dependencies", [&] {
            for (const auto &dependency : source.dependencies)
              json.object([&] {
                json.attribute("module", dependency.first);
                json.attribute("multiplicity", dependency.second);
              });
          });
        });
    });
    json.attributeArray("blocks", [&] {
      for (const auto &block : plan.blocks)
        json.object([&] {
          json.attribute("module", block.module);
          json.attribute("body_lines", block.bodyLines);
          json.attribute("transitive_body_lines", block.transitiveBodyLines);
          json.attribute("root_instance_multiplicity",
                         block.rootInstanceMultiplicity);
          json.attributeArray("source_closure", [&] {
            for (const std::string &module : block.sourceClosureModules)
              writeSourceIdentity(sourceFor(module));
          });
        });
    });
  });
  return output.str();
}

llvm::Expected<external_tool::ResolvedAuxiliaryToolExecutable>
resolveBuildTool(const external_tool::LocalToolConfig &localConfig,
                 llvm::StringRef slot, llvm::ArrayRef<llvm::StringRef> names) {
  const auto canonicalUsable = [](llvm::StringRef candidate)
      -> std::optional<std::string> {
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

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig) {
  const auto &provider = external_tool::verilatorProvider();
  const auto configured = localConfig.tools.find(provider.binding.key);
  if (configured != localConfig.tools.end()) {
    for (const auto &option : configured->second.providerOptions) {
      const llvm::StringRef name = option.first;
      if (name != kCycleLimitOption && name != kBuildJobsOption &&
          name != kBuildWorkersOption && name != kModelThreadsOption)
        return invalid(
            llvm::Twine("verilator.provider_options contains unknown field ") +
            name);
    }
  }
  auto cycleLimit =
      positiveOption(localConfig, kCycleLimitOption, kDefaultCycleLimit,
                     std::numeric_limits<std::uint64_t>::max());
  auto buildJobs =
      positiveOption(localConfig, kBuildJobsOption, mappedRtlDefaultBuildJobs,
                     kMaximumParallelism);
  auto buildWorkers =
      positiveOption(localConfig, kBuildWorkersOption,
                     mappedRtlDefaultBuildWorkers, kMaximumBuildWorkers);
  auto modelThreads =
      positiveOption(localConfig, kModelThreadsOption,
                     mappedRtlDefaultModelThreads, kMaximumParallelism);
  if (!cycleLimit)
    return cycleLimit.takeError();
  if (!buildJobs)
    return buildJobs.takeError();
  if (!isParallelismCount(*buildJobs))
    return invalidParallelism("verilator.provider_options." +
                              kBuildJobsOption);
  if (!buildWorkers)
    return buildWorkers.takeError();
  if (!modelThreads)
    return modelThreads.takeError();
  if (!isParallelismCount(*modelThreads))
    return invalidParallelism("verilator.provider_options." +
                              kModelThreadsOption);
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
  auto hierarchyLauncher =
      resolveBuildTool(localConfig, mappedRtlHierarchyLauncherSlot,
                       {LOOM_MAPPED_RTL_HIERARCHY_LAUNCHER_PATH});
  if (!make || !cxx || !linker || !archiver || !hierarchyLauncher)
    return llvm::joinErrors(
        make ? llvm::Error::success() : make.takeError(),
        llvm::joinErrors(
            cxx ? llvm::Error::success() : cxx.takeError(),
            llvm::joinErrors(
                linker ? llvm::Error::success() : linker.takeError(),
                llvm::joinErrors(archiver ? llvm::Error::success()
                                          : archiver.takeError(),
                                 hierarchyLauncher
                                     ? llvm::Error::success()
                                     : hierarchyLauncher.takeError()))));
  MappedRtlBuildTools result{make->absolutePath,
                             cxx->absolutePath,
                             linker->absolutePath,
                             linker->absolutePath,
                             archiver->absolutePath,
                             hierarchyLauncher->absolutePath,
                             {}};
  if (llvm::StringRef(std::filesystem::path(linker->absolutePath)
                          .filename()
                          .string())
          .starts_with("clang"))
    result.linkerInvocation += " --driver-mode=g++";
  result.provenance.push_back(std::move(*make));
  result.provenance.push_back(std::move(*cxx));
  result.provenance.push_back(std::move(*linker));
  result.provenance.push_back(std::move(*archiver));
  result.provenance.push_back(std::move(*hierarchyLauncher));
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
  if (!isParallelismCount(plan.buildJobs))
    return invalidParallelism("mapped RTL build jobs");
  if (!isParallelismCount(plan.modelThreads))
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
  auto hierarchy = deriveHierarchyPlan(facts, pathPrefix);
  if (!hierarchy)
    return hierarchy.takeError();
  auto prefix = canonicalPathPrefix(pathPrefix);
  if (!prefix)
    return prefix.takeError();
  for (external_tool::MaterializedBundleFile &file : facts.semanticInputs) {
    auto path = namespacedBundlePath(file.relativePath, *prefix);
    if (!path)
      return path.takeError();
    file.relativePath = std::move(*path);
  }
  for (ParsedSource &source : hierarchy->first) {
    auto original = namespacedBundlePath(source.originalPath, *prefix);
    auto path = namespacedBundlePath(source.derivedPath, *prefix);
    if (!original || !path)
      return llvm::joinErrors(original ? llvm::Error::success()
                                       : original.takeError(),
                              path ? llvm::Error::success() : path.takeError());
    source.originalPath = std::move(*original);
    source.derivedPath = std::move(*path);
  }
  if (hierarchy->first.empty())
    return invalid("CIRCT hierarchy has no reachable module source");
  hierarchy->second.sourcePath = hierarchy->first.front().originalPath;
  hierarchy->second.derivedRtlPaths.clear();
  for (const ParsedSource &source : hierarchy->first)
    hierarchy->second.derivedRtlPaths.push_back(source.derivedPath);
  auto rtlLibraryDirectoryPath =
      namespacedBundlePath(hierarchy->second.rtlLibraryDirectoryPath, *prefix);
  auto hierarchyManifestPath =
      namespacedBundlePath(hierarchy->second.manifestPath, *prefix);
  auto workDirectoryPath =
      namespacedBundlePath(hierarchy->second.workDirectoryPath, *prefix);
  if (!rtlLibraryDirectoryPath || !hierarchyManifestPath || !workDirectoryPath)
    return llvm::joinErrors(
        rtlLibraryDirectoryPath ? llvm::Error::success()
                                : rtlLibraryDirectoryPath.takeError(),
        llvm::joinErrors(hierarchyManifestPath
                             ? llvm::Error::success()
                             : hierarchyManifestPath.takeError(),
                         workDirectoryPath ? llvm::Error::success()
                                           : workDirectoryPath.takeError()));
  hierarchy->second.rtlLibraryDirectoryPath =
      std::move(*rtlLibraryDirectoryPath);
  hierarchy->second.manifestPath = std::move(*hierarchyManifestPath);
  hierarchy->second.workDirectoryPath = std::move(*workDirectoryPath);
  facts.rtlPaths.clear();
  facts.rtlLibraryDirectories = {hierarchy->second.rtlLibraryDirectoryPath};
  const std::string hierarchyManifest =
      renderHierarchyPlan(hierarchy->second, hierarchy->first);
  std::vector<external_tool::MaterializedBundleFile> toolLocalInputs;
  toolLocalInputs.reserve(hierarchy->first.size() + 1);
  for (ParsedSource &source : hierarchy->first) {
    toolLocalInputs.push_back({source.derivedPath,
                               std::move(source.derivedBytes),
                               std::nullopt,
                               false});
  }
  toolLocalInputs.push_back(
      {hierarchy->second.manifestPath, hierarchyManifest, std::nullopt,
       false});
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
  if (!testbenchPath || !standaloneDriverPath || !bridgedDriverPath ||
      !bridgeEngineSourcePath || !simulatorExecutablePath || !resultPath)
    return invalid("mapped RTL bundle path namespace is invalid");
  auto testbench = detail::renderMappedRtlTestbench(
      facts, configurationProgramPaths, *resultPath);
  if (!testbench)
    return testbench.takeError();
  // The hierarchy launcher configures child Verilation, which exists only in
  // the hierarchical style.
  std::vector<std::string> launcherVariables;
  if (hierarchy->second.style == MappedRtlVerilationStyle::Hierarchical) {
    auto variables = renderHierarchyLauncherMakeVariables(
        plan.buildTools, plan.verilatorExecutable, *testbenchPath);
    if (!variables)
      return variables.takeError();
    launcherVariables = std::move(*variables);
  }
  auto standalone = detail::renderMappedRtlVerilatorDriver(
      facts, plan, hierarchy->second.style, launcherVariables, *testbenchPath,
      *simulatorExecutablePath, std::nullopt);
  if (!standalone)
    return standalone.takeError();
  auto bridged = detail::renderMappedRtlVerilatorDriver(
      facts, plan, hierarchy->second.style, launcherVariables, *testbenchPath,
      *simulatorExecutablePath, llvm::StringRef(*bridgeEngineSourcePath));
  if (!bridged)
    return bridged.takeError();
  MappedRtlExecutionBundleProjection result;
  result.buildCommand = renderVerilationBuildCommand(
      plan.buildTools, hierarchy->second, plan.buildJobs, launcherVariables);
  result.semanticInputs = std::move(facts.semanticInputs);
  result.toolLocalInputs = std::move(toolLocalInputs);
  result.hierarchy = std::move(hierarchy->second);
  result.configurationProgramPaths = std::move(configurationProgramPaths);
  result.testbenchPath = std::move(*testbenchPath);
  result.standaloneVerilatorDriverPath = std::move(*standaloneDriverPath);
  result.bridgedVerilatorDriverPath = std::move(*bridgedDriverPath);
  result.bridgeEngineSourcePath = std::move(*bridgeEngineSourcePath);
  result.simulatorExecutablePath = std::move(*simulatorExecutablePath);
  result.resultPath = std::move(*resultPath);
  result.testbench = std::move(*testbench);
  result.standaloneVerilatorDriver = std::move(*standalone);
  result.bridgedVerilatorDriver = std::move(*bridged);
  return MappedRtlExecutionProjectionOrUnsupported{std::move(result)};
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

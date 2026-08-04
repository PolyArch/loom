#include "ExternalTool/Binding.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

class RecordingProbe final : public ToolBindingProbe {
public:
  std::map<std::string, ProbedToolBinding> executables;
  std::map<std::string, ProbedToolBinding> modules;
  unsigned executableCalls = 0;
  unsigned moduleCalls = 0;

  llvm::Expected<std::optional<ProbedToolBinding>>
  probeExecutable(llvm::StringRef path) override {
    ++executableCalls;
    auto found = executables.find(path.str());
    if (found == executables.end())
      return std::optional<ProbedToolBinding>{};
    return std::optional<ProbedToolBinding>{found->second};
  }

  llvm::Expected<std::optional<ProbedToolBinding>>
  probeModules(const ModuleProbeRequest &request) override {
    ++moduleCalls;
    std::string key;
    for (llvm::StringRef module : request.modules) {
      if (!key.empty())
        key += ':';
      key += module;
    }
    auto found = modules.find(key);
    if (found == modules.end())
      return std::optional<ProbedToolBinding>{};
    return std::optional<ProbedToolBinding>{found->second};
  }
};

ToolProviderDescriptor descriptor() {
  return ToolProviderDescriptor{
      "verilator",
      {"verilator"},
      {{"VERILATOR_ROOT", "bin/verilator"}},
      {"verilator", "verilator/latest"},
  };
}

ProbedToolBinding probed(llvm::StringRef path, llvm::StringRef version) {
  return ProbedToolBinding{path.str(), version.str(), {}, std::nullopt};
}

void explicitBindingWinsAndFailsClosed() {
  LocalToolConfig config;
  config.tools["verilator"].binding.executable = "/configured/verilator";
  ToolEnvironment environment;
  environment.path = "/environment/bin";
  RecordingProbe probe;
  probe.executables.emplace("/configured/verilator",
                            probed("/configured/verilator", "5.050"));

  ResolvedToolBinding resolved = take(
      __func__, resolveToolBinding(descriptor(), config, environment, probe));
  require(__func__, resolved.source == ToolBindingSource::Explicit,
          "explicit binding did not win");
  require(__func__, resolved.executable == "/configured/verilator",
          "wrong explicit executable");
  require(__func__, probe.executableCalls == 1 && probe.moduleCalls == 0,
          "lower-priority sources were probed");

  probe.executables.clear();
  probe.executableCalls = 0;
  expectErrorContains(
      __func__, resolveToolBinding(descriptor(), config, environment, probe),
      "explicit binding");
  require(__func__, probe.moduleCalls == 0,
          "invalid explicit binding fell through to modules");
}

void environmentPrecedesModules() {
  LocalToolConfig config;
  ToolEnvironment environment;
  environment.path = "/environment/bin";
  RecordingProbe probe;
  probe.executables.emplace("/environment/bin/verilator",
                            probed("/environment/bin/verilator", "5.050"));
  probe.modules.emplace("verilator", ProbedToolBinding{"/module/bin/verilator",
                                                       "5.050",
                                                       {"verilator/5.050"},
                                                       "/module/init/bash"});

  ResolvedToolBinding resolved = take(
      __func__, resolveToolBinding(descriptor(), config, environment, probe));
  require(__func__, resolved.source == ToolBindingSource::EnvironmentPath,
          "PATH binding did not win");
  require(__func__, probe.moduleCalls == 0,
          "module discovery ran after a valid PATH binding");
}

void moduleIsTheFinalFallback() {
  LocalToolConfig config;
  config.moduleInit = "/module/init/bash";
  ToolEnvironment environment;
  RecordingProbe probe;
  probe.modules.emplace("verilator",
                        ProbedToolBinding{"/module/bin/verilator",
                                          "5.050",
                                          {"z3/4.15", "verilator/5.050"},
                                          "/module/init/bash"});

  ResolvedToolBinding resolved = take(
      __func__, resolveToolBinding(descriptor(), config, environment, probe));
  require(__func__, resolved.source == ToolBindingSource::Module,
          "module was not selected as final fallback");
  require(__func__,
          resolved.requestedModules == std::vector<std::string>{"verilator"},
          "requested module alias was not recorded");
  require(__func__,
          resolved.loadedModules ==
              std::vector<std::string>{"z3/4.15", "verilator/5.050"},
          "exact loaded module closure was not frozen");
  require(__func__, probe.moduleCalls == 1,
          "module aliases were probed after a valid result");
}

void moduleProbeMustReturnLoadedClosure() {
  LocalToolConfig config;
  config.moduleInit = "/module/init/bash";
  ToolEnvironment environment;
  RecordingProbe probe;
  probe.modules.emplace("verilator", ProbedToolBinding{"/module/bin/verilator",
                                                       "5.050",
                                                       {},
                                                       "/module/init/bash"});

  expectErrorContains(
      __func__, resolveToolBinding(descriptor(), config, environment, probe),
      "loaded-module closure");
}

void invalidProviderDescriptorsFailBeforeProbing() {
  LocalToolConfig config;
  ToolEnvironment environment;
  RecordingProbe probe;

  ToolProviderDescriptor invalidEnvironment = descriptor();
  invalidEnvironment.environmentCandidates = {{"BAD=ROOT", "bin/verilator"}};
  expectErrorContains(
      __func__,
      resolveToolBinding(invalidEnvironment, config, environment, probe),
      "invalid environment variable name");

  ToolProviderDescriptor invalidRelativePath = descriptor();
  invalidRelativePath.environmentCandidates = {
      {"VERILATOR_ROOT", "../bin/verilator"}};
  expectErrorContains(
      __func__,
      resolveToolBinding(invalidRelativePath, config, environment, probe),
      "invalid relative executable");

  ToolProviderDescriptor invalidModule = descriptor();
  invalidModule.moduleAliases = {""};
  expectErrorContains(
      __func__, resolveToolBinding(invalidModule, config, environment, probe),
      "invalid module alias");

  require(__func__, probe.executableCalls == 0 && probe.moduleCalls == 0,
          "an invalid provider descriptor reached probing");
}

} // namespace

int main() {
  explicitBindingWinsAndFailsClosed();
  environmentPrecedesModules();
  moduleIsTheFinalFallback();
  moduleProbeMustReturnLoadedClosure();
  invalidProviderDescriptorsFailBeforeProbing();
  return 0;
}

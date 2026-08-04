#include "ExternalTool/RuntimeBinding.h"

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
  const std::string message = llvm::toString(value.takeError());
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
    auto found = modules.find(request.modules.front());
    if (found == modules.end())
      return std::optional<ProbedToolBinding>{};
    return std::optional<ProbedToolBinding>{found->second};
  }
};

ResolvedToolBinding toolBinding() {
  return ResolvedToolBinding{"vcs",        ToolBindingSource::EnvironmentPath,
                             "/tools/vcs", "VCS 2025.06",
                             {},           {},
                             std::nullopt, std::nullopt};
}

ToolProviderDescriptor containerDescriptor() {
  return ToolProviderDescriptor{"polyarch_container",
                                {"container"},
                                {{"POLYARCH_CONTAINER_ROOT", "container"}},
                                {"container"}};
}

ProbedToolBinding containerBinding(llvm::StringRef path) {
  return ProbedToolBinding{
      path.str(), "PolyArch container v0.1.0", {}, std::nullopt};
}

void hostPolicyDoesNotResolveContainer() {
  LocalToolConfig config;
  config.runtimePolicy = RuntimePolicy::Host;
  RecordingProbe probe;
  ToolEnvironment environment;
  environment.path = "/environment/bin";
  probe.executables.emplace("/environment/bin/container",
                            containerBinding("/environment/bin/container"));
  ToolRuntimeCompatibility compatibility{true, {"almalinux9"}};

  InvocationRuntimeBinding resolved = take(
      __func__,
      resolveInvocationRuntime(
          toolBinding(), config, containerDescriptor(), environment, probe,
          compatibility,
          [](const ResolvedToolBinding &, const ResolvedToolBinding &,
             llvm::StringRef) -> llvm::Expected<std::optional<std::string>> {
            return std::optional<std::string>{};
          }));
  require(__func__, resolved.kind == InvocationRuntimeKind::Host,
          "host policy did not select host");
  require(__func__, probe.executableCalls == 0 && probe.moduleCalls == 0,
          "host policy probed the container");
}

void autoSelectsFirstCompatibleComposition() {
  LocalToolConfig config;
  ToolEnvironment environment;
  environment.path = "/environment/bin";
  RecordingProbe probe;
  probe.executables.emplace("/environment/bin/container",
                            containerBinding("/environment/bin/container"));
  ToolRuntimeCompatibility compatibility{true, {"almalinux9", "almalinux8"}};
  std::vector<std::string> attempted;

  InvocationRuntimeBinding resolved =
      take(__func__,
           resolveInvocationRuntime(
               toolBinding(), config, containerDescriptor(), environment, probe,
               compatibility,
               [&](const ResolvedToolBinding &tool,
                   const ResolvedToolBinding &container, llvm::StringRef os)
                   -> llvm::Expected<std::optional<std::string>> {
                 require(__func__,
                         tool.toolKey == "vcs" &&
                             container.toolKey == "polyarch_container",
                         "preflight did not receive independent bindings");
                 attempted.push_back(os.str());
                 if (os == "almalinux9")
                   return std::optional<std::string>(
                       "site wrapper conflicts with outer runtime");
                 return std::optional<std::string>{};
               }));
  require(__func__,
          resolved.kind == InvocationRuntimeKind::PolyArchContainer &&
              resolved.os == "almalinux8",
          "auto did not freeze the first compatible composition");
  require(__func__,
          attempted == std::vector<std::string>{"almalinux9", "almalinux8"},
          "OS preferences were not attempted in descriptor order");
  require(__func__, resolved.rejectedCompositions.size() == 1,
          "rejected composition provenance was not retained");
}

void autoFallsBackOnlyWhenContainerWasNotExplicit() {
  ToolRuntimeCompatibility compatibility{true, {"almalinux9"}};
  ToolEnvironment environment;
  RecordingProbe probe;
  LocalToolConfig config;

  InvocationRuntimeBinding host = take(
      __func__,
      resolveInvocationRuntime(
          toolBinding(), config, containerDescriptor(), environment, probe,
          compatibility,
          [](const ResolvedToolBinding &, const ResolvedToolBinding &,
             llvm::StringRef) -> llvm::Expected<std::optional<std::string>> {
            return std::optional<std::string>{};
          }));
  require(__func__,
          host.kind == InvocationRuntimeKind::Host &&
              !host.rejectedCompositions.empty(),
          "auto did not record unavailable container fallback");

  config.polyArchContainer.binding.executable = "/configured/container";
  expectErrorContains(
      __func__,
      resolveInvocationRuntime(
          toolBinding(), config, containerDescriptor(), environment, probe,
          compatibility,
          [](const ResolvedToolBinding &, const ResolvedToolBinding &,
             llvm::StringRef) -> llvm::Expected<std::optional<std::string>> {
            return std::optional<std::string>{};
          }),
      "explicit binding");
  require(__func__, probe.moduleCalls > 0,
          "unconfigured auto resolution did not reach module fallback");
}

void requiredContainerFailsOnRejectedComposition() {
  LocalToolConfig config;
  config.runtimePolicy = RuntimePolicy::PolyArchContainer;
  config.polyArchContainer.binding.executable = "/configured/container";
  config.polyArchContainer.os = "almalinux9";
  ToolEnvironment environment;
  RecordingProbe probe;
  probe.executables.emplace("/configured/container",
                            containerBinding("/configured/container"));
  ToolRuntimeCompatibility compatibility{true, {"almalinux9"}};

  expectErrorContains(
      __func__,
      resolveInvocationRuntime(
          toolBinding(), config, containerDescriptor(), environment, probe,
          compatibility,
          [](const ResolvedToolBinding &, const ResolvedToolBinding &,
             llvm::StringRef) -> llvm::Expected<std::optional<std::string>> {
            return std::optional<std::string>("mount is unavailable");
          }),
      "mount is unavailable");
}

} // namespace

int main() {
  hostPolicyDoesNotResolveContainer();
  autoSelectsFirstCompatibleComposition();
  autoFallsBackOnlyWhenContainerWasNotExplicit();
  requiredContainerFailsOnRejectedComposition();
  return 0;
}

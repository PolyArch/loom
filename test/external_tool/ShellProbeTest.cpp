#include "ExternalTool/ShellProbe.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
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

std::string shellQuote(llvm::StringRef value) {
  std::string result = "'";
  for (char character : value) {
    if (character == '\'')
      result += "'\\''";
    else
      result += character;
  }
  result += "'";
  return result;
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents,
               bool executable = false) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  stream.close();
  if (!stream)
    fail(__func__, "could not write " + path.string());
  if (executable) {
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 std::filesystem::perm_options::replace);
  }
}

std::string toolScript(llvm::StringRef expectedArgument) {
  return "#!/usr/bin/env bash\n"
         "set -eu\n"
         "[[ \"${1-}\" == " +
         shellQuote(expectedArgument) +
         " ]]\n"
         "printf '%s\\n' 'Verilator 5.050'\n";
}

std::string diagnosticToolScript() {
  return "#!/usr/bin/env bash\n"
         "set -u\n"
         "[[ \"${1-}\" == --version ]] || exit 2\n"
         "printf '%s\\n' 'host: changes-between-runtimes'\n"
         "printf '%s\\n' 'dc_shell version - Y-2026.03-SP2'\n"
         "printf '%s\\n' 'timestamp: changes-between-invocations'\n"
         "exit 1\n";
}

std::string moduleInitScript(const std::filesystem::path &binaryDirectory) {
  return "module() {\n"
         "  [[ \"${1-}\" == load ]] || return 2\n"
         "  export PATH=" +
         shellQuote(binaryDirectory.string()) +
         ":\"$PATH\"\n"
         "  export LOADEDMODULES='dependency/1.0:'\"${2-}\"\n"
         "}\n";
}

std::string
moduleInitScriptWithoutClosure(const std::filesystem::path &binaryDirectory) {
  return "module() {\n"
         "  [[ \"${1-}\" == load ]] || return 2\n"
         "  export PATH=" +
         shellQuote(binaryDirectory.string()) +
         ":\"$PATH\"\n"
         "  unset LOADEDMODULES\n"
         "}\n";
}

std::string moduleRootInitScript(const std::filesystem::path &toolRoot) {
  return "module() {\n"
         "  [[ \"${1-}\" == load ]] || return 2\n"
         "  export TOOL_ROOT=" +
         shellQuote(toolRoot.string()) +
         "\n"
         "  export LOADEDMODULES=\"${2-}\"\n"
         "}\n";
}

void explicitInitIsShellSafe(const std::filesystem::path &root) {
  const std::filesystem::path marker = root / "unexpected-command";
  const std::string versionArgument = "--version; touch " + marker.string();
  const std::filesystem::path binaryDirectory =
      root / "bin with spaces; shell data";
  const std::filesystem::path executable = binaryDirectory / "verilator";
  writeFile(executable, toolScript(versionArgument), true);

  const std::filesystem::path init = root / "EL modules init.sh";
  writeFile(init, moduleInitScript(binaryDirectory));

  ShellToolBindingProbe probe(
      root.string(), ToolVersionProbe{{versionArgument}, "Verilator 5.050"});
  ModuleProbeRequest request;
  request.initScript = init.string();
  request.modules = {"verilator; touch " + marker.string()};
  request.executableNames = {"verilator"};

  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeModules(request));
  require(__func__, result.has_value(), "module probe did not bind the tool");
  require(__func__, result->executable == executable.string(),
          "module probe did not freeze the resolved executable");
  require(__func__, result->version == "Verilator 5.050",
          "module probe did not freeze normalized version output");
  require(__func__,
          result->loadedModules ==
              std::vector<std::string>{"dependency/1.0",
                                       "verilator; touch " + marker.string()},
          "module probe did not freeze the loaded module closure");
  require(__func__, result->moduleInit == init.string(),
          "explicit module initialization was not recorded");
  require(__func__, !std::filesystem::exists(marker),
          "shell data was evaluated as a command");
}

void moduleHomeInitIsDiscovered(const std::filesystem::path &root) {
  const std::filesystem::path binaryDirectory = root / "module-home-bin";
  const std::filesystem::path executable = binaryDirectory / "verilator";
  writeFile(executable, toolScript("--version"), true);

  const std::filesystem::path moduleHome = root / "module-home";
  const std::filesystem::path init = moduleHome / "init" / "bash";
  writeFile(init, moduleInitScript(binaryDirectory));

  const char *oldModuleHome = std::getenv("MODULESHOME");
  const std::optional<std::string> savedModuleHome =
      oldModuleHome ? std::optional<std::string>(oldModuleHome) : std::nullopt;

  // Isolate from an inherited module-loaded environment: an exported module
  // function correctly wins over the MODULESHOME fallback this fixture
  // exercises, so the function definitions must leave the process
  // environment before probing and be restored exactly afterwards.
  static constexpr const char *kModuleFunctionVariables[] = {
      "BASH_FUNC_module%%", "BASH_FUNC_ml%%", "BASH_FUNC__module_raw%%"};
  std::vector<std::pair<std::string, std::optional<std::string>>>
      savedModuleFunctions;
  for (const char *name : kModuleFunctionVariables) {
    const char *value = std::getenv(name);
    savedModuleFunctions.emplace_back(
        name, value ? std::optional<std::string>(value) : std::nullopt);
    ::unsetenv(name);
  }

  require(__func__, ::setenv("MODULESHOME", moduleHome.c_str(), 1) == 0,
          "could not set MODULESHOME");

  ShellToolBindingProbe probe(root.string(),
                              ToolVersionProbe{{"--version"}, "Verilator"});
  ModuleProbeRequest request;
  request.modules = {"verilator"};
  request.executableNames = {"verilator"};
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeModules(request));

  if (savedModuleHome)
    ::setenv("MODULESHOME", savedModuleHome->c_str(), 1);
  else
    ::unsetenv("MODULESHOME");
  for (const auto &[name, value] : savedModuleFunctions) {
    if (value)
      ::setenv(name.c_str(), value->c_str(), 1);
    else
      ::unsetenv(name.c_str());
  }

  require(__func__, result.has_value(),
          "$MODULESHOME initialization did not bind the tool");
  require(__func__, result->moduleInit == init.string(),
          "$MODULESHOME initialization path was not recorded");
}

void incompatibleVersionIsRejected(const std::filesystem::path &root) {
  const std::filesystem::path executable = root / "plain" / "verilator";
  writeFile(executable, toolScript("--version"), true);
  ShellToolBindingProbe probe(root.string(),
                              ToolVersionProbe{{"--version"}, "Verilator 6."});
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeExecutable(executable.string()));
  require(__func__, !result, "an incompatible version was accepted");
}

void providerExitAndStableVersionLineAreNormalized(
    const std::filesystem::path &root) {
  const std::filesystem::path executable = root / "bin" / "dc_shell";
  writeFile(executable, diagnosticToolScript(), true);
  ShellToolBindingProbe probe(
      root.string(),
      ToolVersionProbe{
          {"--version"}, "dc_shell version", {0, 1}, "dc_shell version"});
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeExecutable(executable.string()));
  require(__func__, result.has_value(),
          "provider-specific version exit was rejected");
  require(__func__, result->version == "dc_shell version - Y-2026.03-SP2",
          "volatile version output was not reduced to the stable line");
}

std::string invocationNameToolScript() {
  return "#!/usr/bin/env bash\n"
         "set -eu\n"
         "[[ \"${1-}\" == -version ]]\n"
         "printf '%s version 1.0\\n' \"$(basename \"$0\")\"\n";
}

void symlinkedLauncherKeepsItsInvocationName(
    const std::filesystem::path &root) {
  const std::filesystem::path shell = root / "bin" / "suite_shell";
  writeFile(shell, invocationNameToolScript(), true);
  const std::filesystem::path launcher = root / "bin" / "dc_shell";
  std::filesystem::create_symlink("suite_shell", launcher);
  ShellToolBindingProbe probe(
      root.string(),
      ToolVersionProbe{{"-version"}, "dc_shell version", {0}, "version"});
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeExecutable(launcher.string()));
  require(__func__, result.has_value(), "symlinked launcher was not resolved");
  require(__func__, result->executable == launcher.string(),
          "frozen executable lost the probed launcher name");
  require(__func__, result->version == "dc_shell version 1.0",
          "frozen version is not the launcher's own product line");
}

void moduleEnvironmentRootCanSelectLauncher(const std::filesystem::path &root) {
  const std::filesystem::path toolRoot = root / "tool-root";
  const std::filesystem::path executable = toolRoot / "bin" / "tool64";
  writeFile(executable, toolScript("--version"), true);
  const std::filesystem::path init = root / "module-init.sh";
  writeFile(init, moduleRootInitScript(toolRoot));

  ShellToolBindingProbe probe(root.string(),
                              ToolVersionProbe{{"--version"}, "Verilator"});
  ModuleProbeRequest request;
  request.initScript = init.string();
  request.modules = {"vendor/suite"};
  request.executableNames = {"broken-path-launcher"};
  request.environmentCandidates = {{"TOOL_ROOT", "bin/tool64"}};
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeModules(request));
  require(__func__, result.has_value(),
          "module root variable did not resolve a launcher");
  require(__func__, result->executable == executable.string(),
          "module root variable resolved the wrong launcher");
}

void moduleWithoutClosureIsRejected(const std::filesystem::path &root) {
  const std::filesystem::path binaryDirectory = root / "bin";
  writeFile(binaryDirectory / "verilator", toolScript("--version"), true);
  const std::filesystem::path init = root / "module-init.sh";
  writeFile(init, moduleInitScriptWithoutClosure(binaryDirectory));

  ShellToolBindingProbe probe(root.string(),
                              ToolVersionProbe{{"--version"}, "Verilator"});
  ModuleProbeRequest request;
  request.initScript = init.string();
  request.modules = {"verilator"};
  request.executableNames = {"verilator"};
  std::optional<ProbedToolBinding> result =
      take(__func__, probe.probeModules(request));
  require(__func__, !result,
          "a module binding without a loaded-module closure was accepted");
}

void containerCompositionHonorsRequiredEnvironment(
    const std::filesystem::path &root) {
  const std::filesystem::path binaries = root / "bin";
  writeFile(binaries / "fake_tool",
            "#!/usr/bin/env bash\n"
            "if [[ \"${1-}\" == --version ]]; then\n"
            "  printf '%s\\n' 'Fake Tool 3.1'\n"
            "  exit 0\n"
            "fi\n"
            "exit 64\n",
            true);
  writeFile(binaries / "container",
            "#!/usr/bin/env bash\n"
            "set -u\n"
            "[[ \"${1-}\" == run ]] || exit 64\n"
            "shift\n"
            "while (( $# )); do\n"
            "  case \"$1\" in\n"
            "    --workdir|--os|--env) shift 2 ;;\n"
            "    --) shift; break ;;\n"
            "    *) exit 64 ;;\n"
            "  esac\n"
            "done\n"
            "exec \"$@\"\n",
            true);

  const ResolvedToolBinding tool{"fake_tool", ToolBindingSource::Explicit,
                                 (binaries / "fake_tool").string(),
                                 "Fake Tool 3.1",
                                 {},
                                 {},
                                 std::nullopt,
                                 std::nullopt};
  const ResolvedToolBinding container{"polyarch_container",
                                      ToolBindingSource::Explicit,
                                      (binaries / "container").string(),
                                      "PolyArch container v0.1.0",
                                      {},
                                      {},
                                      std::nullopt,
                                      std::nullopt};
  const ToolVersionProbe probe{{"--version"}, "Fake Tool"};

  require(__func__,
          ::setenv("LOOM_COMPOSE_TEST_LICENSE", "present", 1) == 0,
          "could not set the test environment");
  std::optional<std::string> accepted =
      take(__func__, probeContainerToolComposition(
                         root.string(), tool, probe, container, "almalinux9",
                         {"LOOM_COMPOSE_TEST_LICENSE"}));
  require(__func__, !accepted.has_value(),
          std::string("a resolvable composition was rejected") +
              (accepted ? ": " + *accepted : ""));

  require(__func__, ::unsetenv("LOOM_COMPOSE_TEST_LICENSE") == 0,
          "could not unset the test environment");
  std::optional<std::string> rejected =
      take(__func__, probeContainerToolComposition(
                         root.string(), tool, probe, container, "almalinux9",
                         {"LOOM_COMPOSE_TEST_LICENSE"}));
  require(__func__,
          rejected.has_value() &&
              rejected->find("LOOM_COMPOSE_TEST_LICENSE") !=
                  std::string::npos,
          "a missing required environment variable was not rejected");
}

void moduleInitializationLayoutsAreCovered() {
  const llvm::ArrayRef<llvm::StringLiteral> paths =
      defaultModuleInitializationPaths();
  auto contains = [&](llvm::StringRef expected) {
    for (llvm::StringRef path : paths)
      if (path == expected)
        return true;
    return false;
  };
  require(__func__, contains("/etc/profile.d/modules.sh"),
          "common EL module initialization is absent");
  require(__func__, contains("/usr/share/lmod/lmod/init/bash"),
          "common Lmod initialization is absent");
  require(__func__,
          contains("/usr/share/Modules/init/bash") &&
              contains("/usr/share/modules/init/bash"),
          "common Environment Modules layouts are absent");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  explicitInitIsShellSafe(root / "explicit");
  moduleHomeInitIsDiscovered(root / "module-home-discovery");
  incompatibleVersionIsRejected(root / "incompatible");
  providerExitAndStableVersionLineAreNormalized(root / "stable-version");
  symlinkedLauncherKeepsItsInvocationName(root / "symlinked-launcher");
  moduleEnvironmentRootCanSelectLauncher(root / "module-root");
  moduleWithoutClosureIsRejected(root / "missing-module-closure");
  containerCompositionHonorsRequiredEnvironment(root / "composition");
  moduleInitializationLayoutsAreCovered();
  return 0;
}

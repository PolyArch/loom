#include "ExternalTool/ShellProbe.h"

#include "ExternalTool/LocalConfig.h"

#include "ShellRenderingInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace loom::external_tool {
namespace {

using detail::shellQuote;

constexpr int kCandidateUnavailable = 20;

constexpr llvm::StringLiteral kDefaultModuleInitializationPaths[] = {
    "/etc/profile.d/modules.sh",    "/etc/profile.d/lmod.sh",
    "/etc/profile.d/z00_lmod.sh",   "/usr/share/lmod/lmod/init/bash",
    "/usr/share/Modules/init/bash", "/usr/share/modules/init/bash",
};

llvm::Error probeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tool_probe_failed: " + message);
}

llvm::Error validateVersionProbe(const ToolVersionProbe &probe) {
  if (probe.acceptedExitCodes.empty())
    return probeError("version probe has no accepted exit codes");
  std::vector<int> exitCodes = probe.acceptedExitCodes;
  llvm::sort(exitCodes);
  for (std::size_t index = 0; index < exitCodes.size(); ++index) {
    if (exitCodes[index] < 0 || exitCodes[index] > 255)
      return probeError("version probe exit code is outside [0, 255]");
    if (index != 0 && exitCodes[index - 1] == exitCodes[index])
      return probeError("version probe exit codes are not unique");
  }
  if (probe.requiredOutputSubstring && probe.requiredOutputSubstring->empty())
    return probeError("required version output substring is empty");
  if (probe.selectedOutputLineSubstring &&
      probe.selectedOutputLineSubstring->empty())
    return probeError("selected version line substring is empty");
  return llvm::Error::success();
}

/// Collapses every run of blanks to one space so a tool that aligns its
/// version line with tabs freezes the same identity as one that uses spaces;
/// the generated launcher applies the same rule before comparing.
std::string collapseBlankRuns(llvm::StringRef text) {
  std::string collapsed;
  collapsed.reserve(text.size());
  bool pendingBlank = false;
  for (const char character : text) {
    if (character == ' ' || character == '\t') {
      pendingBlank = true;
      continue;
    }
    if (pendingBlank && !collapsed.empty())
      collapsed += ' ';
    pendingBlank = false;
    collapsed += character;
  }
  return collapsed;
}

} // namespace

std::optional<std::string>
normalizeToolVersionOutput(llvm::StringRef versionText,
                           const ToolVersionProbe &probe) {
  std::string version = versionText.trim().str();
  if (version.empty())
    return std::nullopt;
  if (probe.requiredOutputSubstring &&
      !llvm::StringRef(version).contains(*probe.requiredOutputSubstring))
    return std::nullopt;
  if (probe.selectedOutputLineSubstring) {
    llvm::SmallVector<llvm::StringRef, 16> lines;
    llvm::StringRef(version).split(lines, '\n', -1, false);
    std::optional<std::string> selected;
    for (llvm::StringRef line : lines) {
      line = line.trim();
      if (!line.contains(*probe.selectedOutputLineSubstring))
        continue;
      if (selected)
        return std::nullopt;
      selected = line.str();
    }
    if (!selected || selected->empty())
      return std::nullopt;
    version = std::move(*selected);
  }
  return collapseBlankRuns(version);
}

namespace {

struct ProbeFiles {
  std::string script;
  std::string executable;
  std::string version;
  std::string loadedModules;
  std::string moduleInit;

  ProbeFiles() = default;
  ProbeFiles(const ProbeFiles &) = delete;
  ProbeFiles &operator=(const ProbeFiles &) = delete;
  ProbeFiles(ProbeFiles &&other) noexcept
      : script(std::exchange(other.script, {})),
        executable(std::exchange(other.executable, {})),
        version(std::exchange(other.version, {})),
        loadedModules(std::exchange(other.loadedModules, {})),
        moduleInit(std::exchange(other.moduleInit, {})) {}
  ProbeFiles &operator=(ProbeFiles &&) = delete;

  ~ProbeFiles() {
    std::error_code ignored;
    std::filesystem::remove(script, ignored);
    std::filesystem::remove(executable, ignored);
    std::filesystem::remove(version, ignored);
    std::filesystem::remove(loadedModules, ignored);
    std::filesystem::remove(moduleInit, ignored);
  }
};

llvm::Expected<ProbeFiles> createProbeFiles(llvm::StringRef directory) {
  const std::filesystem::path root(directory.str());
  std::error_code statusError;
  if (!root.is_absolute() ||
      !std::filesystem::is_directory(root, statusError) || statusError)
    return probeError("probe directory must be an existing absolute directory");

  llvm::SmallString<256> pattern(directory);
  llvm::sys::path::append(pattern, "loom-tool-probe-%%%%%%.sh");
  int descriptor = -1;
  llvm::SmallString<256> script;
  if (std::error_code error =
          llvm::sys::fs::createUniqueFile(pattern, descriptor, script))
    return probeError("could not create probe script: " + error.message());
  ::close(descriptor);

  ProbeFiles files;
  files.script = script.str().str();
  files.executable = files.script + ".executable";
  files.version = files.script + ".version";
  files.loadedModules = files.script + ".modules";
  files.moduleInit = files.script + ".module-init";
  return files;
}

llvm::Error writeProbeScript(llvm::StringRef path, llvm::StringRef body) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return probeError("could not open probe script: " + error.message());
  output << body;
  output.close();
  if (output.has_error())
    return probeError("could not write probe script");
  return llvm::Error::success();
}

llvm::Expected<std::string> readProbeFile(llvm::StringRef path,
                                          llvm::StringRef field) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return probeError("could not read " + field + ": " +
                      buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

llvm::Expected<bool> executeProbeScript(const ProbeFiles &files) {
  llvm::ErrorOr<std::string> bash = llvm::sys::findProgramByName("bash");
  if (!bash)
    return probeError("could not find bash: " + bash.getError().message());
  const llvm::SmallVector<llvm::StringRef, 2> arguments{*bash, files.script};
  std::string message;
  bool executionFailed = false;
  const int status = llvm::sys::ExecuteAndWait(
      *bash, arguments, std::nullopt, {}, 0, 0, &message, &executionFailed);
  if (status == 0)
    return true;
  if (status == kCandidateUnavailable)
    return false;
  if (executionFailed || status < 0)
    return probeError("could not execute generated probe script: " + message);
  return probeError("generated probe script exited with status " +
                    llvm::Twine(status));
}

std::string renderVersionInvocation(llvm::StringRef executableExpression,
                                    const ToolVersionProbe &probe,
                                    llvm::StringRef versionPath) {
  std::string command = executableExpression.str();
  for (const std::string &argument : probe.arguments)
    command += " " + shellQuote(argument);
  command += " >" + shellQuote(versionPath) + " 2>&1\n";
  command += "loom_version_status=$?\n";
  command += "case \"$loom_version_status\" in\n";
  for (int exitCode : probe.acceptedExitCodes)
    command += "  " + std::to_string(exitCode) + ") ;;\n";
  command += "  *) exit " + std::to_string(kCandidateUnavailable) + " ;;\n";
  command += "esac\n";
  return command;
}

std::string renderExecutableProbe(const ProbeFiles &files, llvm::StringRef path,
                                  const ToolVersionProbe &probe) {
  std::string script = "#!/usr/bin/env bash\nset -u\n";
  script += "loom_executable=" + shellQuote(path) + "\n";
  script += "[[ -x \"$loom_executable\" ]] || exit " +
            std::to_string(kCandidateUnavailable) + "\n";
  script +=
      renderVersionInvocation("\"$loom_executable\"", probe, files.version);
  script += "printf '%s' \"$loom_executable\" >" +
            shellQuote(files.executable) + "\n";
  return script;
}

void appendModuleInitAttempt(std::string &script, llvm::StringRef candidate) {
  script += "elif loom_try_module_init " + shellQuote(candidate) + "; then\n";
  script += "  :\n";
}

std::string renderModuleProbe(const ProbeFiles &files,
                              const ModuleProbeRequest &request,
                              const ToolVersionProbe &probe) {
  std::string script =
      "#!/usr/bin/env bash\n"
      "set -u\n"
      "loom_try_module_init() {\n"
      "  local loom_candidate=\"$1\"\n"
      "  [[ -r \"$loom_candidate\" ]] || return 1\n"
      "  source \"$loom_candidate\" >/dev/null 2>&1 || return 1\n"
      "  type module >/dev/null 2>&1 || return 1\n"
      "  printf '%s' \"$loom_candidate\" >" +
      shellQuote(files.moduleInit) +
      "\n"
      "}\n";

  if (request.initScript) {
    script += "if loom_try_module_init " + shellQuote(*request.initScript) +
              "; then\n  :\nelse\n  exit " +
              std::to_string(kCandidateUnavailable) + "\nfi\n";
  } else {
    script += "if type module >/dev/null 2>&1; then\n";
    script += "  : >" + shellQuote(files.moduleInit) + "\n";
    script += "elif [[ -n \"${MODULESHOME:-}\" ]] && "
              "loom_try_module_init \"${MODULESHOME}/init/bash\"; then\n"
              "  :\n";
    for (llvm::StringRef candidate : defaultModuleInitializationPaths())
      appendModuleInitAttempt(script, candidate);
    script +=
        "else\n  exit " + std::to_string(kCandidateUnavailable) + "\nfi\n";
  }

  for (const std::string &module : request.modules) {
    script +=
        "if ! module load " + shellQuote(module) + " >/dev/null 2>&1; then\n";
    script += "  exit " + std::to_string(kCandidateUnavailable) + "\nfi\n";
  }

  script += "loom_executable=''\n";
  for (const ToolEnvironmentCandidate &candidate :
       request.environmentCandidates) {
    script += "if [[ -z \"$loom_executable\" ]]; then\n";
    script += "  loom_root=$(printenv " + shellQuote(candidate.variable) +
              " 2>/dev/null || true)\n";
    script += "  if [[ -n \"$loom_root\" ]]; then\n";
    script += "    loom_candidate=\"$loom_root\"/" +
              shellQuote(candidate.relativeExecutable) + "\n";
    script += "    if [[ -x \"$loom_candidate\" ]]; then "
              "loom_executable=\"$loom_candidate\"; fi\n";
    script += "  fi\n";
    script += "fi\n";
  }
  for (const std::string &name : request.executableNames) {
    script += "if [[ -z \"$loom_executable\" ]]; then\n";
    script += "  loom_candidate=$(type -P " + shellQuote(name) +
              " 2>/dev/null || true)\n";
    script += "  if [[ -n \"$loom_candidate\" && -x \"$loom_candidate\" ]]; "
              "then loom_executable=\"$loom_candidate\"; fi\n";
    script += "fi\n";
  }
  script += "[[ -n \"$loom_executable\" ]] || exit " +
            std::to_string(kCandidateUnavailable) + "\n";
  script +=
      renderVersionInvocation("\"$loom_executable\"", probe, files.version);
  script += "printf '%s' \"$loom_executable\" >" +
            shellQuote(files.executable) + "\n";
  script += "printf '%s' \"${LOADEDMODULES:-}\" >" +
            shellQuote(files.loadedModules) + "\n";
  return script;
}

llvm::Expected<std::optional<ProbedToolBinding>>
collectProbeResult(const ProbeFiles &files, const ToolVersionProbe &probe,
                   bool includeModuleState) {
  llvm::Expected<std::string> executableText =
      readProbeFile(files.executable, "resolved executable");
  if (!executableText)
    return executableText.takeError();
  // Canonicalize the directory while retaining the probed launcher name.
  // Suite launchers dispatch on argv[0], so resolving the final symlink would
  // freeze a different program identity.
  const llvm::StringRef executable(*executableText);
  llvm::SmallString<256> canonicalExecutable;
  if (std::error_code error = llvm::sys::fs::real_path(
          llvm::sys::path::parent_path(executable), canonicalExecutable, true))
    return probeError("could not canonicalize resolved executable: " +
                      error.message());
  llvm::sys::path::append(canonicalExecutable,
                          llvm::sys::path::filename(executable));

  llvm::Expected<std::string> versionText =
      readProbeFile(files.version, "version output");
  if (!versionText)
    return versionText.takeError();
  std::optional<std::string> normalized =
      normalizeToolVersionOutput(*versionText, probe);
  if (!normalized)
    return std::optional<ProbedToolBinding>{};

  ProbedToolBinding binding;
  binding.executable = canonicalExecutable.str().str();
  binding.version = std::move(*normalized);
  if (!includeModuleState)
    return std::optional<ProbedToolBinding>{std::move(binding)};

  llvm::Expected<std::string> modulesText =
      readProbeFile(files.loadedModules, "loaded module closure");
  if (!modulesText)
    return modulesText.takeError();
  llvm::SmallVector<llvm::StringRef, 8> modules;
  llvm::StringRef(*modulesText).split(modules, ':', -1, false);
  if (modules.empty())
    return std::optional<ProbedToolBinding>{};
  for (llvm::StringRef module : modules)
    binding.loadedModules.push_back(module.str());

  llvm::Expected<std::string> initText =
      readProbeFile(files.moduleInit, "module initialization path");
  if (!initText)
    return initText.takeError();
  if (!initText->empty())
    binding.moduleInit = std::move(*initText);
  return std::optional<ProbedToolBinding>{std::move(binding)};
}

} // namespace

llvm::ArrayRef<llvm::StringLiteral> defaultModuleInitializationPaths() {
  return kDefaultModuleInitializationPaths;
}

ShellToolBindingProbe::ShellToolBindingProbe(std::string probeDirectory,
                                             ToolVersionProbe versionProbe)
    : probeDirectory_(std::move(probeDirectory)),
      versionProbe_(std::move(versionProbe)) {}

llvm::Expected<std::optional<ProbedToolBinding>>
ShellToolBindingProbe::probeExecutable(llvm::StringRef path) {
  if (llvm::Error error = validateVersionProbe(versionProbe_))
    return std::move(error);
  llvm::Expected<ProbeFiles> files = createProbeFiles(probeDirectory_);
  if (!files)
    return files.takeError();
  if (llvm::Error error = writeProbeScript(
          files->script, renderExecutableProbe(*files, path, versionProbe_)))
    return std::move(error);
  llvm::Expected<bool> succeeded = executeProbeScript(*files);
  if (!succeeded)
    return succeeded.takeError();
  if (!*succeeded)
    return std::optional<ProbedToolBinding>{};
  return collectProbeResult(*files, versionProbe_, false);
}

llvm::Expected<std::optional<ProbedToolBinding>>
ShellToolBindingProbe::probeModules(const ModuleProbeRequest &request) {
  if (request.modules.empty() || request.executableNames.empty())
    return probeError("module probe requires modules and executable names");
  if (llvm::Error error = validateVersionProbe(versionProbe_))
    return std::move(error);
  llvm::Expected<ProbeFiles> files = createProbeFiles(probeDirectory_);
  if (!files)
    return files.takeError();
  if (llvm::Error error = writeProbeScript(
          files->script, renderModuleProbe(*files, request, versionProbe_)))
    return std::move(error);
  llvm::Expected<bool> succeeded = executeProbeScript(*files);
  if (!succeeded)
    return succeeded.takeError();
  if (!*succeeded)
    return std::optional<ProbedToolBinding>{};
  return collectProbeResult(*files, versionProbe_, true);
}

llvm::Expected<std::optional<std::string>> probeContainerToolComposition(
    llvm::StringRef probeDirectory, const ResolvedToolBinding &tool,
    const ToolVersionProbe &toolVersionProbe,
    const ResolvedToolBinding &polyArchContainer, llvm::StringRef os,
    llvm::ArrayRef<std::string> inheritEnvironment) {
  if (llvm::Error error = validateVersionProbe(toolVersionProbe))
    return std::move(error);
  for (const std::string &name : inheritEnvironment) {
    if (!isValidEnvironmentName(name))
      return probeError("required environment name is invalid");
    if (!std::getenv(name.c_str()))
      return std::optional<std::string>(
          "container composition lacks required environment variable " + name);
  }
  llvm::Expected<ProbeFiles> files = createProbeFiles(probeDirectory);
  if (!files)
    return files.takeError();

  std::vector<std::string> arguments{
      "/usr/bin/bash", "-c",
      "loom_version_path=$1\nshift\n\"$@\" >\"$loom_version_path\" 2>&1",
      "loom-container-version", files->version, tool.executable};
  arguments.insert(arguments.end(), toolVersionProbe.arguments.begin(),
                   toolVersionProbe.arguments.end());

  std::string script = "#!/usr/bin/env bash\nset -u\n";
  script += detail::renderPolyArchContainerInvocation(
      polyArchContainer.executable, os, detail::shellQuote(probeDirectory),
      arguments);
  script += "\nloom_status=$?\n";
  script += "case \"$loom_status\" in\n";
  for (int exitCode : toolVersionProbe.acceptedExitCodes)
    script += "  " + std::to_string(exitCode) + ") ;;\n";
  script += "  *) exit " + std::to_string(kCandidateUnavailable) + " ;;\n";
  script += "esac\n";

  if (llvm::Error error = writeProbeScript(files->script, script))
    return std::move(error);
  llvm::Expected<bool> succeeded = executeProbeScript(*files);
  if (!succeeded)
    return succeeded.takeError();
  if (!*succeeded)
    return std::optional<std::string>(
        "container composition could not execute the tool version probe");
  llvm::Expected<std::string> versionText =
      readProbeFile(files->version, "version output");
  if (!versionText)
    return versionText.takeError();
  std::optional<std::string> normalized =
      normalizeToolVersionOutput(*versionText, toolVersionProbe);
  if (!normalized)
    return std::optional<std::string>(
        "container composition version output did not match the probe");
  if (*normalized != tool.version)
    return std::optional<std::string>(
        "container composition version does not match the resolved tool");
  return std::optional<std::string>{};
}

} // namespace loom::external_tool

#include "InvocationBundleInternal.h"

#include "ShellRenderingInternal.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace loom::external_tool {
namespace {

using detail::shellQuote;

constexpr std::uint64_t maximumCommandStatusBytes = 4;
constexpr std::uint64_t maximumCommandTimingBytes = 64 * 1024;
constexpr std::uint64_t maximumCommandStreamBytes = 1024 * 1024 * 1024;

constexpr int exitCode(InvocationLauncherExitCode code) {
  return static_cast<int>(code);
}

std::string exitCodeText(InvocationLauncherExitCode code) {
  return std::to_string(exitCode(code));
}

std::string renderContainerInvocation(const std::vector<std::string> &arguments,
                                      const InvocationManifestData &manifest) {
  return detail::renderPolyArchContainerInvocation(
      manifest.runtime.polyArchContainer->executable, *manifest.runtime.os,
      "\"$loom_bundle_root\"", arguments);
}

std::string renderCommand(const std::vector<std::string> &command,
                          const InvocationManifestData &manifest) {
  if (manifest.runtime.kind == InvocationRuntimeKind::PolyArchContainer) {
    std::vector<std::string> containerArguments{
        "/usr/bin/bash", "-c",
        "cd -- \"$HOME/work\" || exit " +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) +
            "\nexec \"$@\"",
        "loom-container-entry"};
    containerArguments.insert(containerArguments.end(), command.begin(),
                              command.end());
    return renderContainerInvocation(containerArguments, manifest);
  }
  std::string rendered;
  for (const std::string &argument : command)
    rendered += (rendered.empty() ? "" : " ") + shellQuote(argument);
  return rendered;
}

std::string renderDirectCommand(llvm::StringRef executable,
                                const ToolVersionProbe &probe) {
  std::string rendered = shellQuote(executable);
  for (const std::string &argument : probe.arguments)
    rendered += " " + shellQuote(argument);
  return rendered;
}

std::vector<std::string> frozenModules(const InvocationManifestData &manifest) {
  std::vector<std::string> modules;
  auto append = [&](const ResolvedToolBinding &binding) {
    for (const std::string &module : binding.loadedModules)
      if (std::find(modules.begin(), modules.end(), module) == modules.end())
        modules.push_back(module);
  };
  if (manifest.runtime.polyArchContainer)
    append(*manifest.runtime.polyArchContainer);
  append(manifest.tool);
  return modules;
}

std::optional<std::string>
moduleInitializer(const InvocationManifestData &manifest) {
  if (manifest.tool.moduleInit)
    return manifest.tool.moduleInit;
  if (manifest.runtime.polyArchContainer)
    return manifest.runtime.polyArchContainer->moduleInit;
  return std::nullopt;
}

void appendFailure(std::string &script, InvocationCompletionStatus status,
                   InvocationLauncherExitCode code) {
  script += "  if (( loom_cache_postflight == 0 )); then\n";
  script += "    loom_publish_completion " +
            shellQuote(completionStatusSpelling(status)) + " " +
            exitCodeText(code) + "\n";
  script += "  fi\n";
  script += "  exit " + exitCodeText(code) + "\n";
}

void appendContentDigestCheck(std::string &script, llvm::StringRef path,
                              llvm::StringRef expectedDigest) {
  script += "loom_digest=''\n";
  script += "if ! IFS= read -r -N 64 loom_digest < <(sha256sum --zero -- " +
            shellQuote(path) +
            " 2>/dev/null) || "
            "[[ \"$loom_digest\" != " +
            shellQuote(expectedDigest) + " ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
}

void appendOrdinaryDirectoryPathCheck(std::string &script,
                                      const std::filesystem::path &path) {
  std::filesystem::path current;
  for (const std::filesystem::path &component : path) {
    current /= component;
    const std::string spelling = current.generic_string();
    script += "if [[ ! -d " + shellQuote(spelling) + " || -L " +
              shellQuote(spelling) + " ]]; then\n";
    appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                  InvocationLauncherExitCode::BundleContentMismatch);
    script += "fi\n";
  }
}

void appendFileTreeCheck(std::string &script,
                         const ResolvedExternalFileTree &tree) {
  script += "if [[ ! -d " + shellQuote(tree.absolutePath) + " || -L " +
            shellQuote(tree.absolutePath) + " ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "loom_tree_special=$(find -P -- " + shellQuote(tree.absolutePath) +
            " ! -type d ! -type f -print -quit 2>/dev/null)\n";
  script += "loom_status=$?\n";
  script += "if (( loom_status != 0 )) || [[ -n \"$loom_tree_special\" ]]; "
            "then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "loom_tree_count=$(find -P -- " + shellQuote(tree.absolutePath) +
            " -type f -printf . 2>/dev/null | wc -c)\n";
  script += "loom_status=$?\n";
  script += "if (( loom_status != 0 )) || [[ \"$loom_tree_count\" != " +
            shellQuote(std::to_string(tree.members.size())) + " ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  for (const ExternalFileTreeMember &member : tree.members) {
    const std::filesystem::path path =
        std::filesystem::path(tree.absolutePath) / member.relativePath;
    appendContentDigestCheck(script, path.string(),
                             formatExternalFileFingerprint(member.fingerprint));
  }
}

void appendVersionStatusCheck(std::string &script,
                              const ToolVersionProbe &probe) {
  script += "if ! (( ";
  for (std::size_t index = 0; index < probe.acceptedExitCodes.size(); ++index) {
    if (index != 0)
      script += " || ";
    script +=
        "loom_status == " + std::to_string(probe.acceptedExitCodes[index]);
  }
  script += " )); then\n";
  appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                InvocationLauncherExitCode::VersionMismatch);
  script += "fi\n";
}

void appendVersionOutputCheck(std::string &script,
                              const ToolVersionProbe &probe,
                              llvm::StringRef expected) {
  if (probe.selectedOutputLineSubstring) {
    script += "loom_version_selector=" +
              shellQuote(*probe.selectedOutputLineSubstring) + "\n";
    script += "loom_version_selected=''\n";
    script += "loom_version_match_count=0\n";
    script += "while IFS= read -r loom_version_line; do\n";
    script += "  if [[ \"$loom_version_line\" == "
              "*\"$loom_version_selector\"* ]]; then\n";
    script += "    loom_version_selected=\"$loom_version_line\"\n";
    script += "    (( loom_version_match_count += 1 ))\n";
    script += "  fi\n";
    script += "done <<< \"$loom_version_output\"\n";
    script += "if (( loom_version_match_count != 1 )); then\n";
    appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                  InvocationLauncherExitCode::VersionMismatch);
    script += "fi\n";
    script += "loom_version_output=\"$loom_version_selected\"\n";
  }
  script += "while [[ \"$loom_version_output\" == [[:space:]]* ]]; do "
            "loom_version_output=\"${loom_version_output:1}\"; done\n";
  script += "while [[ \"$loom_version_output\" == *[[:space:]] ]]; do "
            "loom_version_output=\"${loom_version_output:0:${#loom_version_"
            "output}-1}\"; done\n";
  script += "if [[ \"$loom_version_output\" != " + shellQuote(expected) +
            " ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                InvocationLauncherExitCode::VersionMismatch);
  script += "fi\n";
}

void appendVersionCheck(std::string &script, llvm::StringRef command,
                        const ToolVersionProbe &probe,
                        llvm::StringRef expected) {
  script += "loom_version_output=$(" + command.str() + " 2>&1)\n";
  script += "loom_status=$?\n";
  appendVersionStatusCheck(script, probe);
  appendVersionOutputCheck(script, probe, expected);
}

void appendContainerToolVersionCheck(std::string &script,
                                     const InvocationManifestData &manifest) {
  std::vector<std::string> arguments{
      "/usr/bin/bash",
      "-c",
      "cd -- \"$HOME/work\" || exit " +
          exitCodeText(InvocationLauncherExitCode::LauncherFailure) +
          "\nloom_version_path=$1\nshift\n"
          "\"$@\" >\"$loom_version_path\" 2>&1",
      "loom-container-version",
      kToolVersionPath.str(),
      manifest.tool.executable,
  };
  arguments.insert(arguments.end(), manifest.toolVersionProbe.arguments.begin(),
                   manifest.toolVersionProbe.arguments.end());
  script += "loom_tool_version_file=" + shellQuote(kToolVersionPath) + "\n";
  script += "rm -f -- \"$loom_tool_version_file\"\n";
  script +=
      renderContainerInvocation(arguments, manifest) + " >/dev/null 2>&1\n";
  script += "loom_status=$?\n";
  appendVersionStatusCheck(script, manifest.toolVersionProbe);
  script += "if [[ ! -f \"$loom_tool_version_file\" ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                InvocationLauncherExitCode::VersionMismatch);
  script += "fi\n";
  script += "loom_version_output=$(<\"$loom_tool_version_file\")\n";
  script += "rm -f -- \"$loom_tool_version_file\"\n";
  script += "loom_tool_version_file=''\n";
  appendVersionOutputCheck(script, manifest.toolVersionProbe,
                           manifest.tool.version);
}

std::string commandExecutionStem(std::uint64_t ordinal) {
  return kCommandExecutionDirectory.str() + "/" + std::to_string(ordinal);
}

std::string commandDescriptorVariable(std::uint64_t ordinal,
                                      llvm::StringRef role) {
  return "loom_command_" + std::to_string(ordinal) + "_" + role.str() + "_fd";
}

void appendCommandScratchPreparation(std::string &script,
                                     std::uint64_t ordinal) {
  const std::string stem = commandExecutionStem(ordinal);
  script += "  if ! exec {" +
            commandDescriptorVariable(ordinal, "stdout_write") + "}>" +
            shellQuote(stem + ".stdout") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "stdout_read") + "}<" +
            shellQuote(stem + ".stdout") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "stderr_write") + "}>" +
            shellQuote(stem + ".stderr") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "stderr_read") + "}<" +
            shellQuote(stem + ".stderr") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "wall_write") + "}>" +
            shellQuote(stem + ".wall") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "wall_read") + "}<" +
            shellQuote(stem + ".wall") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "status_write") + "}>" +
            shellQuote(stem + ".status") + " || ! exec {" +
            commandDescriptorVariable(ordinal, "status_read") + "}<" +
            shellQuote(stem + ".status") + "; then\n";
  script += "    loom_status=" +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) + "\n";
  script += "    loom_schedule_infrastructure_failure=1\n";
  script += "  fi\n";
}

void appendCommandDescriptorsClosedForTool(std::string &script,
                                           std::uint64_t begin,
                                           std::uint64_t end) {
  script += "        exec {loom_stdout_fd}>&-\n";
  script += "        exec {loom_stderr_fd}>&-\n";
  script += "        exec {loom_command_observations_fd}>&-\n";
  static constexpr std::array<llvm::StringLiteral, 8> roles{
      "stdout_write", "stdout_read", "stderr_write", "stderr_read",
      "wall_write",   "wall_read",   "status_write", "status_read"};
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    for (const llvm::StringRef role : roles)
      script += "        exec {" + commandDescriptorVariable(ordinal, role) +
                "}>&-\n";
}

void appendProducedExecutableChecks(std::string &script,
                                    const std::vector<std::string> &command,
                                    const InvocationManifestData &manifest) {
  if (!llvm::is_contained(manifest.toolProducedExecutables, command.front()))
    return;
  std::vector<std::string> referencedExecutables{command.front()};
  for (auto argument = std::next(command.begin()); argument != command.end();
       ++argument)
    if (llvm::is_contained(manifest.toolProducedExecutables, *argument) &&
        !llvm::is_contained(referencedExecutables, *argument))
      referencedExecutables.push_back(*argument);
  for (const std::string &executable : referencedExecutables) {
    script += "if [[ -L " + shellQuote(executable) + " ]]; then\n";
    script += "  loom_publish_completion " +
              shellQuote(completionStatusSpelling(
                  InvocationCompletionStatus::BundleContentMismatch)) +
              " " +
              exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
              "\n";
    script += "  exit " +
              exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
              "\n";
    script += "elif [[ ! -f " + shellQuote(executable) + " || ! -x " +
              shellQuote(executable) + " ]]; then\n";
    script +=
        "  loom_status=" +
        exitCodeText(
            InvocationLauncherExitCode::ToolProducedExecutableUnavailable) +
        "\n";
    script += "fi\n";
  }
}

void appendCommandLaunch(std::string &script,
                         const InvocationManifestData &manifest,
                         std::uint64_t ordinal, std::uint64_t chunkBegin,
                         std::uint64_t chunkEnd) {
  script += "  (\n";
  script += "    loom_command_exit=0\n";
  script += "    loom_command_lc_all_set=${LC_ALL+x}\n";
  script += "    loom_command_lc_all=${LC_ALL-}\n";
  script += "    loom_command_lc_numeric_set=${LC_NUMERIC+x}\n";
  script += "    loom_command_lc_numeric=${LC_NUMERIC-}\n";
  script += "    LC_ALL=C\n";
  script += "    export LC_ALL\n";
  script += "    { TIMEFORMAT='%R'; time {\n";
  script += "      (\n";
  script += "        if [[ \"$loom_command_lc_all_set\" == x ]]; then "
            "LC_ALL=$loom_command_lc_all; export LC_ALL; else unset LC_ALL; "
            "fi\n";
  script += "        if [[ \"$loom_command_lc_numeric_set\" == x ]]; then "
            "LC_NUMERIC=$loom_command_lc_numeric; export LC_NUMERIC; else "
            "unset LC_NUMERIC; fi\n";
  appendCommandDescriptorsClosedForTool(script, chunkBegin, chunkEnd);
  script += "        exec " +
            renderCommand(manifest.commands[ordinal], manifest) + "\n";
  script += "      ) >&\"${" +
            commandDescriptorVariable(ordinal, "stdout_write") + "}\" 2>&\"${" +
            commandDescriptorVariable(ordinal, "stderr_write") + "}\"\n";
  script += "    }; } 2>&\"${" +
            commandDescriptorVariable(ordinal, "wall_write") +
            "}\" || loom_command_exit=$?\n";
  script +=
      "    if ! printf '%s\\n' \"$loom_command_exit\" >&\"${" +
      commandDescriptorVariable(ordinal, "status_write") + "}\"; then exit " +
      exitCodeText(InvocationLauncherExitCode::LauncherFailure) + "; fi\n";
  script += "  ) &\n";
  script += "  loom_command_pids+=(\"$!\")\n";
}

void appendCommandWriterClosure(std::string &script, std::uint64_t ordinal) {
  script +=
      "  if ! exec {" + commandDescriptorVariable(ordinal, "stdout_write") +
      "}>&- || ! exec {" + commandDescriptorVariable(ordinal, "stderr_write") +
      "}>&- || ! exec {" + commandDescriptorVariable(ordinal, "wall_write") +
      "}>&- || ! exec {" + commandDescriptorVariable(ordinal, "status_write") +
      "}>&-; then\n";
  script += "    loom_status=" +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) + "\n";
  script += "    loom_schedule_infrastructure_failure=1\n";
  script += "  fi\n";
}

void appendCommandCollection(std::string &script, std::uint64_t ordinal) {
  script += "  loom_command_exit=''\n";
  script += "  loom_command_wall=''\n";
  script += "  loom_command_wall_text=''\n";
  script += "  loom_command_collection_failed=0\n";
  script += "  if ! loom_command_exit=$(loom_read_command_text \"${" +
            commandDescriptorVariable(ordinal, "status_read") + "}\" " +
            std::to_string(maximumCommandStatusBytes) +
            "); then loom_command_collection_failed=1; fi\n";
  script += "  if ! loom_command_wall_text=$(loom_read_command_text \"${" +
            commandDescriptorVariable(ordinal, "wall_read") + "}\" " +
            std::to_string(maximumCommandTimingBytes) +
            "); then loom_command_collection_failed=1; fi\n";
  script += "  if ! loom_copy_command_stream \"${" +
            commandDescriptorVariable(ordinal, "stdout_read") +
            "}\" \"$loom_stdout_fd\" " +
            std::to_string(maximumCommandStreamBytes) +
            "; then loom_command_collection_failed=1; fi\n";
  script += "  if ! loom_copy_command_stream \"${" +
            commandDescriptorVariable(ordinal, "stderr_read") +
            "}\" \"$loom_stderr_fd\" " +
            std::to_string(maximumCommandStreamBytes) +
            "; then loom_command_collection_failed=1; fi\n";
  script += "  if [[ ! \"$loom_command_exit\" =~ ^[0-9]+$ ]] || "
            "(( loom_command_exit > 255 )); then\n";
  script += "    loom_command_collection_failed=1\n";
  script += "  else\n";
  script += "    loom_command_time_seen=0\n";
  script += "    while IFS= read -r loom_command_time_line; do\n";
  script += "      if (( loom_command_time_seen != 0 )) && ! printf '%s\\n' "
            "\"$loom_command_wall\" >&\"$loom_stderr_fd\"; then\n";
  script += "        loom_command_collection_failed=1\n";
  script += "      fi\n";
  script += "      loom_command_wall=$loom_command_time_line\n";
  script += "      loom_command_time_seen=1\n";
  script += "    done <<<\"$loom_command_wall_text\"\n";
  script += "    if (( loom_command_time_seen == 0 )) || "
            "[[ ! \"$loom_command_wall\" =~ ^[0-9]+\\.[0-9]+$ ]]; then\n";
  script += "      loom_command_collection_failed=1\n";
  script += "    fi\n";
  script += "  fi\n";
  script +=
      "  if ! exec {" + commandDescriptorVariable(ordinal, "stdout_read") +
      "}<&- || ! exec {" + commandDescriptorVariable(ordinal, "stderr_read") +
      "}<&- || ! exec {" + commandDescriptorVariable(ordinal, "wall_read") +
      "}<&- || ! exec {" + commandDescriptorVariable(ordinal, "status_read") +
      "}<&-; then loom_command_collection_failed=1; fi\n";
  script += "  if (( loom_command_collection_failed != 0 )); then\n";
  script += "    loom_status=" +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) + "\n";
  script += "    loom_schedule_infrastructure_failure=1\n";
  script += "  elif ! printf 'command %s %s %s\\n' " +
            shellQuote(std::to_string(ordinal)) +
            " \"$loom_command_wall\" \"$loom_command_exit\" "
            ">&\"$loom_command_observations_fd\"; then\n";
  script += "    loom_status=" +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) + "\n";
  script += "    loom_schedule_infrastructure_failure=1\n";
  script += "  elif (( loom_status == 0 && loom_command_exit != 0 )); then\n";
  script += "    loom_status=$loom_command_exit\n";
  script += "  fi\n";
}

void appendCommandChunk(std::string &script,
                        const InvocationManifestData &manifest,
                        std::uint64_t begin, std::uint64_t end) {
  script += "if (( loom_status == 0 )); then\n";
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    appendProducedExecutableChecks(script, manifest.commands[ordinal],
                                   manifest);
  script += "if (( loom_status == 0 )); then\n";
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    appendCommandScratchPreparation(script, ordinal);
  script += "if (( loom_status == 0 )); then\n";
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    appendCommandLaunch(script, manifest, ordinal, begin, end);
  script += "  for loom_command_pid in \"${loom_command_pids[@]}\"; do\n";
  script += "    if ! wait \"$loom_command_pid\"; then loom_status=" +
            exitCodeText(InvocationLauncherExitCode::LauncherFailure) +
            "; loom_schedule_infrastructure_failure=1; fi\n";
  script += "  done\n";
  script += "  loom_command_pids=()\n";
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    appendCommandWriterClosure(script, ordinal);
  for (std::uint64_t ordinal = begin; ordinal != end; ++ordinal)
    appendCommandCollection(script, ordinal);
  script += "fi\n";
  script += "fi\n";
  script += "fi\n";
}

void appendCommandSchedule(std::string &script,
                           const InvocationManifestData &manifest) {
  std::size_t groupOrdinal = 0;
  std::uint64_t commandOrdinal = 0;
  while (commandOrdinal != manifest.commands.size()) {
    if (groupOrdinal == manifest.parallelCommandGroups.size() ||
        manifest.parallelCommandGroups[groupOrdinal].beginCommandOrdinal !=
            commandOrdinal) {
      appendCommandChunk(script, manifest, commandOrdinal, commandOrdinal + 1);
      ++commandOrdinal;
      continue;
    }
    const ExternalToolParallelCommandGroup &group =
        manifest.parallelCommandGroups[groupOrdinal++];
    for (std::uint64_t begin = group.beginCommandOrdinal;
         begin != group.endCommandOrdinal;) {
      const std::uint64_t end =
          std::min(group.endCommandOrdinal, begin + group.workerLimit);
      appendCommandChunk(script, manifest, begin, end);
      begin = end;
    }
    commandOrdinal = group.endCommandOrdinal;
  }
}

} // namespace

std::string renderRunScript(const InvocationManifestData &manifest) {
  const std::string manifestDigest =
      formatBlobDigestHex(contentDigest(serializeManifest(manifest)));
  const std::string launcherFailure =
      exitCodeText(InvocationLauncherExitCode::LauncherFailure);
  std::string script = "#!/usr/bin/env bash\n"
                       "set -u -o pipefail\n"
                       "loom_cache_preflight=0\n"
                       "loom_cache_postflight=0\n"
                       "if (( $# == 1 )) && [[ \"$1\" == "
                       "--loom-cache-preflight ]]; then\n"
                       "  loom_cache_preflight=1\n"
                       "elif (( $# == 1 )) && [[ \"$1\" == "
                       "--loom-cache-postflight ]]; then\n"
                       "  loom_cache_postflight=1\n"
                       "elif (( $# != 0 )); then\n"
                       "  exit " +
                       launcherFailure +
                       "\n"
                       "fi\n"
                       "loom_bundle_root=$(CDPATH= cd -- \"$(dirname -- "
                       "\"${BASH_SOURCE[0]}\")\" && pwd -P)\n";
  script += "cd -- \"$loom_bundle_root\" || exit " + launcherFailure + "\n";
  script +=
      "loom_completion=" + shellQuote(kCompletionPath) +
      "\n"
      "loom_completion_partial=\"${loom_completion}.partial.$$\"\n"
      "loom_tool_version_file=''\n"
      "loom_command_observations=''\n"
      "loom_command_observations_partial=''\n"
      "loom_command_execution_directory=''\n"
      "loom_command_pids=()\n"
      "loom_stdout_fd=''\n"
      "loom_stderr_fd=''\n"
      "loom_command_observations_fd=''\n"
      "loom_manifest_digest=" +
      shellQuote(manifestDigest) +
      "\n"
      "loom_attempt_token=''\n"
      "loom_output_digests='[]'\n"
      "loom_cleanup() {\n"
      "  for loom_command_pid in \"${loom_command_pids[@]}\"; do "
      "kill \"$loom_command_pid\" 2>/dev/null || true; done\n"
      "  for loom_command_pid in \"${loom_command_pids[@]}\"; do "
      "wait \"$loom_command_pid\" 2>/dev/null || true; done\n"
      "  rm -f -- \"$loom_completion_partial\"\n"
      "  if [[ -n \"$loom_tool_version_file\" ]]; then rm -f -- "
      "\"$loom_tool_version_file\"; fi\n"
      "  if [[ -n \"$loom_command_observations_partial\" ]]; then rm -f -- "
      "\"$loom_command_observations_partial\"; fi\n"
      "  if [[ -n \"$loom_command_execution_directory\" ]]; then rm -rf -- "
      "\"$loom_command_execution_directory\"; fi\n"
      "}\n"
      "trap loom_cleanup EXIT\n"
      "loom_publish_completion() {\n"
      "  printf "
      "'{\"schema\":\"" +
      kInvocationCompletionSchema.str() + "\",\"version\":\"" +
      kInvocationCompletionVersion.str() +
      "\","
      "\"status\":\"%s\",\"exit_code\":%s,"
      "\"manifest_sha256\":\"%s\",\"attempt_sha256\":\"%s\","
      "\"output_sha256\":%s}\\n' "
      "\"$1\" \"$2\" \"$loom_manifest_digest\" \"$loom_attempt_token\" "
      "\"$loom_output_digests\" "
      ">\"$loom_completion_partial\" || exit " +
      launcherFailure + "\n";
  script += "  mv -f -- \"$loom_completion_partial\" \"$loom_completion\" "
            "|| exit " +
            launcherFailure + "\n}\n";
  script += "if [[ ! -f " + shellQuote(kAttemptTokenPath) + " || -L " +
            shellQuote(kAttemptTokenPath) + " ]]; then exit " +
            exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
            "; fi\n";
  script += "loom_attempt_token=$(<" + shellQuote(kAttemptTokenPath) + ")\n";
  script += "if [[ ! \"$loom_attempt_token\" =~ ^[0-9a-f]{64}$ ]]; then "
            "exit " +
            exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
            "; fi\n";
  script +=
      "loom_read_command_text() {\n"
      "  local loom_read_fd=$1\n"
      "  local loom_read_limit=$2\n"
      "  local loom_size_before=''\n"
      "  local loom_size_after=''\n"
      "  [[ -f /proc/$$/fd/$loom_read_fd ]] || return 1\n"
      "  loom_size_before=$(stat -Lc '%s' -- "
      "\"/proc/$$/fd/$loom_read_fd\") || return 1\n"
      "  [[ \"$loom_size_before\" =~ ^[0-9]+$ ]] || return 1\n"
      "  (( loom_size_before <= loom_read_limit )) || return 1\n"
      "  dd iflag=fullblock,count_bytes,nonblock count=\"$loom_size_before\" "
      "status=none <&\"$loom_read_fd\" || return 1\n"
      "  loom_size_after=$(stat -Lc '%s' -- "
      "\"/proc/$$/fd/$loom_read_fd\") || return 1\n"
      "  [[ \"$loom_size_after\" == \"$loom_size_before\" ]]\n"
      "}\n"
      "loom_copy_command_stream() {\n"
      "  local loom_read_fd=$1\n"
      "  local loom_write_fd=$2\n"
      "  local loom_read_limit=$3\n"
      "  local loom_size_before=''\n"
      "  local loom_size_after=''\n"
      "  [[ -f /proc/$$/fd/$loom_read_fd ]] || return 1\n"
      "  loom_size_before=$(stat -Lc '%s' -- "
      "\"/proc/$$/fd/$loom_read_fd\") || return 1\n"
      "  [[ \"$loom_size_before\" =~ ^[0-9]+$ ]] || return 1\n"
      "  (( loom_size_before <= loom_read_limit )) || return 1\n"
      "  dd iflag=fullblock,count_bytes,nonblock count=\"$loom_size_before\" "
      "status=none <&\"$loom_read_fd\" >&\"$loom_write_fd\" || return 1\n"
      "  loom_size_after=$(stat -Lc '%s' -- "
      "\"/proc/$$/fd/$loom_read_fd\") || return 1\n"
      "  [[ \"$loom_size_after\" == \"$loom_size_before\" ]]\n"
      "}\n";
  script += "if (( loom_cache_preflight == 0 && "
            "loom_cache_postflight == 0 )); then\n";
  script +=
      "  loom_command_observations=" + shellQuote(kCommandObservationsPath) +
      "\n";
  script += "  loom_command_observations_partial=\"${loom_command_"
            "observations}.partial.$$\"\n";
  script += "  if ! rm -f -- \"$loom_command_observations\" "
            "\"$loom_command_observations_partial\" || "
            "! printf 'loom.external_tool_command_observations 1.0\\n"
            "manifest %s\\nattempt %s\\nend\\n' \"$loom_manifest_digest\" "
            "\"$loom_attempt_token\" >\"$loom_command_observations_partial\" "
            "|| ! mv -f -- \"$loom_command_observations_partial\" "
            "\"$loom_command_observations\"; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "  fi\n";
  script += "fi\n";
  // Every removal whose success is required for fresh publication is
  // checked: stale completion, partial, or declared output material that
  // cannot be removed fails integrity before the tool is entered.
  script += "if (( loom_cache_postflight == 0 )); then\n";
  script += "if ! rm -f -- \"$loom_completion\"; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\nif ! rm -f -- \"$loom_completion_partial\"; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "fi\n";

  script += "if ! command -v cat >/dev/null 2>&1 || "
            "! command -v sha256sum >/dev/null 2>&1 || "
            "! command -v find >/dev/null 2>&1 || "
            "! command -v wc >/dev/null 2>&1 || "
            "! command -v dd >/dev/null 2>&1 || "
            "! command -v stat >/dev/null 2>&1; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  appendContentDigestCheck(script, kManifestName, manifestDigest);
  for (const ManifestMaterializedFile &file : manifest.materializedFiles)
    appendContentDigestCheck(script, file.relativePath,
                             formatBlobDigestHex(file.contentDigest));
  for (const ResolvedExternalFile &file : manifest.externalFiles)
    appendContentDigestCheck(script, file.absolutePath,
                             formatExternalFileFingerprint(file.fingerprint));
  for (const ResolvedExternalFileTree &tree : manifest.externalFileTrees)
    appendFileTreeCheck(script, tree);
  if (!manifest.declaredOutputs.empty()) {
    script += "if (( loom_cache_postflight == 0 )); then\n";
    for (const std::string &output : manifest.declaredOutputs) {
      script += "if ! rm -f -- " + shellQuote(output) + "; then\n";
      appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                    InvocationLauncherExitCode::BundleContentMismatch);
      script += "fi\n";
    }
    script += "fi\n";
  }
  std::set<std::filesystem::path> producedExecutableParents;
  for (const std::string &executable : manifest.toolProducedExecutables)
    producedExecutableParents.insert(
        std::filesystem::path(executable).parent_path());
  for (const std::filesystem::path &parent : producedExecutableParents)
    appendOrdinaryDirectoryPathCheck(script, parent);
  if (!manifest.toolProducedExecutables.empty()) {
    script += "if (( loom_cache_postflight == 0 )); then\n";
    for (const std::string &executable : manifest.toolProducedExecutables) {
      script += "if ! rm -f -- " + shellQuote(executable) + "; then\n";
      appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                    InvocationLauncherExitCode::BundleContentMismatch);
      script += "fi\n";
    }
    script += "fi\n";
  }

  for (const std::string &name : manifest.inheritEnvironment) {
    script += "if [[ -z \"${" + name + "+x}\" ]]; then\n";
    appendFailure(script, InvocationCompletionStatus::MissingEnvironment,
                  InvocationLauncherExitCode::MissingEnvironment);
    script += "fi\n";
  }

  const std::vector<std::string> modules = frozenModules(manifest);
  if (!modules.empty()) {
    if (std::optional<std::string> init = moduleInitializer(manifest)) {
      script += "if [[ ! -r " + shellQuote(*init) + " ]] || ! source " +
                shellQuote(*init) + " >/dev/null 2>&1; then\n";
      appendFailure(script, InvocationCompletionStatus::ModuleActivationFailed,
                    InvocationLauncherExitCode::ModuleActivationFailed);
      script += "fi\n";
    }
    script += "if ! type module >/dev/null 2>&1; then\n";
    appendFailure(script, InvocationCompletionStatus::ModuleActivationFailed,
                  InvocationLauncherExitCode::ModuleActivationFailed);
    script += "fi\n";
    for (const std::string &module : modules) {
      script +=
          "if ! module load " + shellQuote(module) + " >/dev/null 2>&1; then\n";
      appendFailure(script, InvocationCompletionStatus::ModuleActivationFailed,
                    InvocationLauncherExitCode::ModuleActivationFailed);
      script += "fi\n";
    }
  }

  script +=
      "if [[ ! -x " + shellQuote(manifest.tool.executable) + " ]]; then\n";
  appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                InvocationLauncherExitCode::VersionMismatch);
  script += "fi\n";
  if (manifest.runtime.polyArchContainer) {
    script += "if [[ ! -x " +
              shellQuote(manifest.runtime.polyArchContainer->executable) +
              " ]]; then\n";
    appendFailure(script, InvocationCompletionStatus::VersionMismatch,
                  InvocationLauncherExitCode::VersionMismatch);
    script += "fi\n";
    appendVersionCheck(
        script,
        renderDirectCommand(manifest.runtime.polyArchContainer->executable,
                            manifest.containerVersionProbe),
        manifest.containerVersionProbe,
        manifest.runtime.polyArchContainer->version);
  }
  if (manifest.runtime.kind == InvocationRuntimeKind::PolyArchContainer)
    appendContainerToolVersionCheck(script, manifest);
  else
    appendVersionCheck(script,
                       renderDirectCommand(manifest.tool.executable,
                                           manifest.toolVersionProbe),
                       manifest.toolVersionProbe, manifest.tool.version);

  script += "if (( loom_cache_preflight != 0 || "
            "loom_cache_postflight != 0 )); then exit 0; fi\n";

  script += "loom_command_execution_directory=" +
            shellQuote(kCommandExecutionDirectory) + "\n";
  script += "if ! rm -rf -- \"$loom_command_execution_directory\" || "
            "! mkdir -p -- \"$loom_command_execution_directory\" || "
            "! exec {loom_stdout_fd}>" +
            shellQuote(kStdoutPath) + " || ! exec {loom_stderr_fd}>" +
            shellQuote(kStderrPath) + "; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "if ! exec {loom_command_observations_fd}>"
            "\"$loom_command_observations_partial\" || ! printf "
            "'loom.external_tool_command_observations 1.0\\n"
            "manifest %s\\nattempt %s\\n' \"$loom_manifest_digest\" "
            "\"$loom_attempt_token\" "
            ">&\"$loom_command_observations_fd\"; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "loom_status=0\n";
  script += "loom_schedule_infrastructure_failure=0\n";
  appendCommandSchedule(script, manifest);
  script += "if ! exec {loom_stdout_fd}>&- || "
            "! exec {loom_stderr_fd}>&- || "
            "! printf 'end\\n' >&\"$loom_command_observations_fd\" || "
            "! exec {loom_command_observations_fd}>&- "
            "|| ! mv -f -- \"$loom_command_observations_partial\" "
            "\"$loom_command_observations\"; then\n";
  appendFailure(script, InvocationCompletionStatus::BundleContentMismatch,
                InvocationLauncherExitCode::BundleContentMismatch);
  script += "fi\n";
  script += "if (( loom_status != 0 )); then\n";
  script += "  if (( loom_schedule_infrastructure_failure != 0 )); then\n";
  script += "    loom_publish_completion " +
            shellQuote(completionStatusSpelling(
                InvocationCompletionStatus::BundleContentMismatch)) +
            " \"$loom_status\"\n";
  script += "  else\n";
  script += "    loom_publish_completion " +
            shellQuote(completionStatusSpelling(
                InvocationCompletionStatus::ToolExit)) +
            " \"$loom_status\"\n";
  script += "  fi\n";
  script += "  exit \"$loom_status\"\n";
  script += "fi\n";
  script += "loom_success_output_digests='['\n";
  for (std::size_t index = 0; index < manifest.declaredOutputs.size();
       ++index) {
    const std::string &output = manifest.declaredOutputs[index];
    script += "if [[ ! -f " + shellQuote(output) + " || -L " +
              shellQuote(output) + " ]]; then\n";
    appendFailure(script, InvocationCompletionStatus::MissingOutput,
                  InvocationLauncherExitCode::MissingOutput);
    script += "fi\n";
    script += "loom_output_digest=''\n";
    script +=
        "if ! IFS= read -r -N 64 loom_output_digest < <(sha256sum --zero -- " +
        shellQuote(output) + " 2>/dev/null); then\n";
    appendFailure(script, InvocationCompletionStatus::MissingOutput,
                  InvocationLauncherExitCode::MissingOutput);
    script += "fi\n";
    if (index != 0)
      script += "loom_success_output_digests+=','\n";
    script +=
        "loom_success_output_digests+=\"\\\"${loom_output_digest}\\\"\"\n";
  }
  script += "loom_success_output_digests+=']'\n";
  script += "loom_output_digests=\"$loom_success_output_digests\"\n";
  script += "loom_publish_completion " +
            shellQuote(
                completionStatusSpelling(InvocationCompletionStatus::Success)) +
            " 0\n";
  return script;
}

} // namespace loom::external_tool

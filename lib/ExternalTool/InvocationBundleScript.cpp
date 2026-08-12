#include "InvocationBundleInternal.h"

#include "ShellRenderingInternal.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace loom::external_tool {
namespace {

using detail::shellQuote;

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
      "loom_manifest_digest=" +
      shellQuote(manifestDigest) +
      "\n"
      "loom_output_digests='[]'\n"
      "trap 'rm -f -- \"$loom_completion_partial\"; if [[ -n "
      "\"$loom_tool_version_file\" ]]; then rm -f -- "
      "\"$loom_tool_version_file\"; fi' EXIT\n"
      "loom_publish_completion() {\n"
      "  printf "
      "'{\"schema\":\"loom.external_tool_completion\",\"version\":\"1.0\","
      "\"status\":\"%s\",\"exit_code\":%s,"
      "\"manifest_sha256\":\"%s\",\"output_sha256\":%s}\\n' "
      "\"$1\" \"$2\" \"$loom_manifest_digest\" "
      "\"$loom_output_digests\" "
      ">\"$loom_completion_partial\" || exit " +
      launcherFailure + "\n";
  script += "  mv -f -- \"$loom_completion_partial\" \"$loom_completion\" "
            "|| exit " +
            launcherFailure + "\n}\n";
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

  script += "if ! command -v sha256sum >/dev/null 2>&1 || "
            "! command -v find >/dev/null 2>&1 || "
            "! command -v wc >/dev/null 2>&1; then\n";
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

  script += "loom_status=0\n";
  script += "{\n";
  for (const std::vector<std::string> &command : manifest.commands) {
    script += "  if (( loom_status == 0 )); then\n";
    const bool generatedController =
        std::find(manifest.toolProducedExecutables.begin(),
                  manifest.toolProducedExecutables.end(),
                  command.front()) != manifest.toolProducedExecutables.end();
    if (generatedController) {
      std::vector<std::string> referencedExecutables{command.front()};
      for (auto argument = std::next(command.begin());
           argument != command.end(); ++argument)
        if (std::find(manifest.toolProducedExecutables.begin(),
                      manifest.toolProducedExecutables.end(),
                      *argument) != manifest.toolProducedExecutables.end() &&
            std::find(referencedExecutables.begin(),
                      referencedExecutables.end(),
                      *argument) == referencedExecutables.end())
          referencedExecutables.push_back(*argument);
      for (const std::string &executable : referencedExecutables) {
        script += "    if [[ -L " + shellQuote(executable) + " ]]; then\n";
        script +=
            "      loom_publish_completion " +
            shellQuote(completionStatusSpelling(
                InvocationCompletionStatus::BundleContentMismatch)) +
            " " +
            exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
            "\n";
        script +=
            "      exit " +
            exitCodeText(InvocationLauncherExitCode::BundleContentMismatch) +
            "\n";
        script += "    elif [[ ! -f " + shellQuote(executable) + " || ! -x " +
                  shellQuote(executable) + " ]]; then\n";
        script +=
            "      loom_status=" +
            exitCodeText(
                InvocationLauncherExitCode::ToolProducedExecutableUnavailable) +
            "\n";
        script += "    fi\n";
      }
      script += "    if (( loom_status == 0 )); then\n";
      script +=
          "      " + renderCommand(command, manifest) + " || loom_status=$?\n";
      script += "    fi\n";
    } else {
      script +=
          "    " + renderCommand(command, manifest) + " || loom_status=$?\n";
    }
    script += "  fi\n";
  }
  script +=
      "} >" + shellQuote(kStdoutPath) + " 2>" + shellQuote(kStderrPath) + "\n";
  script += "if (( loom_status != 0 )); then\n";
  script += "  loom_publish_completion " +
            shellQuote(completionStatusSpelling(
                InvocationCompletionStatus::ToolExit)) +
            " \"$loom_status\"\n";
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

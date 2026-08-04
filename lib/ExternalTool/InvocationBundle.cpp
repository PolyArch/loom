#include "ExternalTool/InvocationBundle.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::external_tool {
namespace {

constexpr llvm::StringLiteral kManifestName = "tool-invocation.json";
constexpr llvm::StringLiteral kRunScriptName = "run.sh";
constexpr llvm::StringLiteral kCompletionPath = "outputs/completion.json";
constexpr llvm::StringLiteral kStdoutPath = "outputs/stdout.log";
constexpr llvm::StringLiteral kStderrPath = "outputs/stderr.log";
constexpr llvm::StringLiteral kToolVersionPath = "outputs/.loom-tool-version";

llvm::Error bundleError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invocation_bundle_invalid: " + message);
}

bool containsNull(llvm::StringRef value) {
  return value.find('\0') != llvm::StringRef::npos;
}

llvm::Expected<std::string> normalizedRelativePath(llvm::StringRef spelling,
                                                   llvm::StringRef field) {
  if (spelling.empty() || containsNull(spelling))
    return bundleError(field + " path is empty or contains NUL");
  const std::filesystem::path path(spelling.str());
  if (path.is_absolute())
    return bundleError(field + " path must be relative");
  for (const std::filesystem::path &component : path)
    if (component == "..")
      return bundleError(field + " path may not contain '..'");
  const std::filesystem::path normalized = path.lexically_normal();
  if (normalized.empty() || normalized == "." ||
      normalized.generic_string() != spelling)
    return bundleError(field + " path must be lexically normalized");
  return normalized.generic_string();
}

bool isWithin(llvm::StringRef path, llvm::StringRef directory) {
  return path.starts_with(directory) && path.size() > directory.size() &&
         path[directory.size()] == '/';
}

llvm::Error reservePath(std::set<std::string> &paths,
                        llvm::StringRef candidate) {
  for (const std::string &existing : paths) {
    if (candidate == existing || isWithin(candidate, existing) ||
        isWithin(existing, candidate))
      return bundleError("bundle paths conflict: '" + candidate + "' and '" +
                         existing + "'");
  }
  paths.insert(candidate.str());
  return llvm::Error::success();
}

llvm::StringRef bindingSourceName(ToolBindingSource source) {
  switch (source) {
  case ToolBindingSource::Explicit:
    return "explicit";
  case ToolBindingSource::EnvironmentPath:
    return "environment_path";
  case ToolBindingSource::EnvironmentRoot:
    return "environment_root";
  case ToolBindingSource::Module:
    return "module";
  }
  llvm_unreachable("unknown tool binding source");
}

std::optional<InvocationCompletionStatus>
parseCompletionStatus(llvm::StringRef spelling) {
  if (spelling == "success")
    return InvocationCompletionStatus::Success;
  if (spelling == "missing_environment")
    return InvocationCompletionStatus::MissingEnvironment;
  if (spelling == "module_activation_failed")
    return InvocationCompletionStatus::ModuleActivationFailed;
  if (spelling == "version_mismatch")
    return InvocationCompletionStatus::VersionMismatch;
  if (spelling == "bundle_content_mismatch")
    return InvocationCompletionStatus::BundleContentMismatch;
  if (spelling == "tool_exit")
    return InvocationCompletionStatus::ToolExit;
  if (spelling == "missing_output")
    return InvocationCompletionStatus::MissingOutput;
  return std::nullopt;
}

llvm::Error validateBinding(const ResolvedToolBinding &binding,
                            llvm::StringRef field) {
  if (binding.toolKey.empty() || containsNull(binding.toolKey))
    return bundleError(field + " tool key is empty or contains NUL");
  if (binding.executable.empty() || containsNull(binding.executable) ||
      !llvm::sys::path::is_absolute(binding.executable))
    return bundleError(field + " executable must be an absolute path");
  if (binding.version.empty() || containsNull(binding.version))
    return bundleError(field + " version is empty or contains NUL");
  if (binding.moduleInit &&
      (containsNull(*binding.moduleInit) ||
       !llvm::sys::path::is_absolute(*binding.moduleInit)))
    return bundleError(field + " module initialization must be absolute");
  for (const std::string &module : binding.requestedModules)
    if (module.empty() || containsNull(module))
      return bundleError(field + " requested module is invalid");
  for (const std::string &module : binding.loadedModules)
    if (module.empty() || containsNull(module))
      return bundleError(field + " loaded module is invalid");
  if (!binding.requestedModules.empty() && binding.loadedModules.empty())
    return bundleError(field + " has an empty loaded-module closure");
  return llvm::Error::success();
}

llvm::Error validateVersionProbe(const ToolVersionProbe &probe,
                                 llvm::StringRef field) {
  for (const std::string &argument : probe.arguments)
    if (containsNull(argument))
      return bundleError(field + " version argument contains NUL");
  if (probe.requiredOutputSubstring &&
      (probe.requiredOutputSubstring->empty() ||
       containsNull(*probe.requiredOutputSubstring)))
    return bundleError(field + " required version output is invalid");
  if (probe.acceptedExitCodes.empty())
    return bundleError(field + " version probe has no accepted exit codes");
  std::vector<int> exitCodes = probe.acceptedExitCodes;
  llvm::sort(exitCodes);
  for (std::size_t index = 0; index < exitCodes.size(); ++index) {
    if (exitCodes[index] < 0 || exitCodes[index] > 255)
      return bundleError(field + " version exit code is outside [0, 255]");
    if (index != 0 && exitCodes[index - 1] == exitCodes[index])
      return bundleError(field + " version exit codes are not unique");
  }
  if (probe.selectedOutputLineSubstring &&
      (probe.selectedOutputLineSubstring->empty() ||
       containsNull(*probe.selectedOutputLineSubstring)))
    return bundleError(field + " selected version line is invalid");
  return llvm::Error::success();
}

llvm::Error
validateSpecification(const ExternalToolInvocationBundleSpec &specification) {
  const std::string *identities[] = {
      &specification.providerIdentity,
      &specification.semanticBindingIdentity,
      &specification.resultImporterIdentity,
  };
  for (const std::string *identity : identities)
    if (identity->empty() || containsNull(*identity))
      return bundleError("bundle identity is empty or contains NUL");
  if (llvm::Error error = validateBinding(specification.tool, "tool binding"))
    return error;
  if (llvm::Error error =
          validateVersionProbe(specification.toolVersionProbe, "tool"))
    return error;

  if (specification.runtime.kind == InvocationRuntimeKind::Host) {
    if (specification.runtime.polyArchContainer || specification.runtime.os)
      return bundleError("host runtime may not contain a container binding");
  } else {
    if (!specification.runtime.polyArchContainer || !specification.runtime.os ||
        specification.runtime.os->empty() ||
        containsNull(*specification.runtime.os))
      return bundleError(
          "PolyArch/container runtime requires a binding and OS");
    if (llvm::Error error = validateBinding(
            *specification.runtime.polyArchContainer, "container binding"))
      return error;
    if (llvm::Error error = validateVersionProbe(
            specification.containerVersionProbe, "container"))
      return error;
    const auto &container = *specification.runtime.polyArchContainer;
    if (container.toolKey != "polyarch_container")
      return bundleError("container binding has the wrong logical tool key");
    if (specification.tool.moduleInit && container.moduleInit &&
        specification.tool.moduleInit != container.moduleInit)
      return bundleError(
          "tool and container bindings use different module initializers");
  }
  for (const std::string &rejection :
       specification.runtime.rejectedCompositions)
    if (rejection.empty() || containsNull(rejection))
      return bundleError("runtime rejection provenance is invalid");

  if (specification.commands.empty())
    return bundleError("bundle has no commands");
  for (const std::vector<std::string> &command : specification.commands) {
    if (command.empty() || command.front() != specification.tool.executable)
      return bundleError(
          "each command must begin with the frozen tool executable");
    for (const std::string &argument : command)
      if (containsNull(argument))
        return bundleError("command argument contains NUL");
  }

  std::set<std::string> environmentNames;
  for (const std::string &name : specification.inheritEnvironment)
    if (!isValidEnvironmentName(name) || !environmentNames.insert(name).second)
      return bundleError("inherited environment name is invalid or duplicated");

  std::set<std::string> externalSlots;
  for (const ResolvedExternalFile &file : specification.externalFiles) {
    if (file.providerInputSlot.empty() ||
        containsNull(file.providerInputSlot) ||
        !externalSlots.insert(file.providerInputSlot).second)
      return bundleError(
          "external file provider input slot is invalid or duplicated");
    if (file.localFileKey.empty() || containsNull(file.localFileKey))
      return bundleError("external file local key is invalid");
    if (file.absolutePath.empty() || containsNull(file.absolutePath))
      return bundleError("external file path is empty or contains NUL");
    const std::filesystem::path path(file.absolutePath);
    if (!path.is_absolute() || path.lexically_normal() != path)
      return bundleError(
          "external file path must be an absolute canonical path");
  }

  std::set<std::string> paths{kManifestName.str(),   kRunScriptName.str(),
                              kCompletionPath.str(), kStdoutPath.str(),
                              kStderrPath.str(),     kToolVersionPath.str()};
  for (const MaterializedBundleFile &file : specification.files) {
    llvm::Expected<std::string> path =
        normalizedRelativePath(file.relativePath, "materialized file");
    if (!path)
      return path.takeError();
    const bool isDriver = isWithin(*path, "drivers");
    const bool isInput = isWithin(*path, "inputs");
    if (!isDriver && !isInput)
      return bundleError("materialized files must be under drivers or inputs");
    if (llvm::Error error = reservePath(paths, *path))
      return error;
    if (file.sourceArtifactIdentity &&
        (file.sourceArtifactIdentity->empty() ||
         containsNull(*file.sourceArtifactIdentity)))
      return bundleError("source Artifact identity is invalid");
    if (isInput && !file.sourceArtifactIdentity)
      return bundleError("materialized input lacks a source Artifact identity");
    if (isDriver && file.sourceArtifactIdentity)
      return bundleError("generated driver may not claim a source Artifact");
  }
  for (const std::string &spelling : specification.declaredOutputs) {
    llvm::Expected<std::string> path =
        normalizedRelativePath(spelling, "declared output");
    if (!path)
      return path.takeError();
    if (!isWithin(*path, "outputs"))
      return bundleError("declared outputs must be under outputs");
    if (llvm::Error error = reservePath(paths, *path))
      return error;
  }
  return llvm::Error::success();
}

void writeStringArray(llvm::json::OStream &json,
                      const std::vector<std::string> &values) {
  json.array([&] {
    for (const std::string &value : values)
      json.value(value);
  });
}

void writeBinding(llvm::json::OStream &json,
                  const ResolvedToolBinding &binding) {
  json.object([&] {
    json.attribute("tool_key", binding.toolKey);
    json.attribute("source", bindingSourceName(binding.source));
    json.attribute("executable", binding.executable);
    json.attribute("version", binding.version);
    json.attributeBegin("requested_modules");
    writeStringArray(json, binding.requestedModules);
    json.attributeEnd();
    json.attributeBegin("loaded_modules");
    writeStringArray(json, binding.loadedModules);
    json.attributeEnd();
    if (binding.moduleInit)
      json.attribute("module_init", *binding.moduleInit);
    if (binding.environmentVariable)
      json.attribute("environment_variable", *binding.environmentVariable);
  });
}

void writeVersionProbe(llvm::json::OStream &json,
                       const ToolVersionProbe &probe) {
  json.object([&] {
    json.attributeBegin("arguments");
    writeStringArray(json, probe.arguments);
    json.attributeEnd();
    if (probe.requiredOutputSubstring)
      json.attribute("required_output_substring",
                     *probe.requiredOutputSubstring);
    json.attributeArray("accepted_exit_codes", [&] {
      for (int exitCode : probe.acceptedExitCodes)
        json.value(exitCode);
    });
    if (probe.selectedOutputLineSubstring)
      json.attribute("selected_output_line_substring",
                     *probe.selectedOutputLineSubstring);
  });
}

std::string contentDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return formatBlobDigestHex(
      computeBlobDigest(llvm::ArrayRef<std::uint8_t>(bytes, contents.size())));
}

std::string
serializeManifest(const ExternalToolInvocationBundleSpec &specification) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", "loom.external_tool_invocation");
    json.attribute("version", "1.0");
    json.attribute("provider_identity", specification.providerIdentity);
    json.attribute("semantic_binding_identity",
                   specification.semanticBindingIdentity);
    json.attribute("result_importer_identity",
                   specification.resultImporterIdentity);
    json.attributeBegin("tool_binding");
    writeBinding(json, specification.tool);
    json.attributeEnd();
    json.attributeBegin("tool_version_probe");
    writeVersionProbe(json, specification.toolVersionProbe);
    json.attributeEnd();
    json.attributeObject("runtime_binding", [&] {
      if (specification.runtime.kind == InvocationRuntimeKind::Host) {
        json.attribute("kind", "host");
      } else {
        json.attribute("kind", "polyarch_container");
        json.attribute("os", *specification.runtime.os);
        json.attributeBegin("container_binding");
        writeBinding(json, *specification.runtime.polyArchContainer);
        json.attributeEnd();
        json.attributeBegin("container_version_probe");
        writeVersionProbe(json, specification.containerVersionProbe);
        json.attributeEnd();
      }
      json.attributeBegin("rejected_compositions");
      writeStringArray(json, specification.runtime.rejectedCompositions);
      json.attributeEnd();
    });
    json.attributeArray("commands", [&] {
      for (const std::vector<std::string> &command : specification.commands)
        writeStringArray(json, command);
    });
    json.attributeBegin("inherit_environment");
    writeStringArray(json, specification.inheritEnvironment);
    json.attributeEnd();
    json.attributeArray("materialized_files", [&] {
      for (const MaterializedBundleFile &file : specification.files) {
        json.object([&] {
          json.attribute("path", file.relativePath);
          json.attribute("executable", file.executable);
          json.attribute("content_sha256", contentDigest(file.contents));
          if (file.sourceArtifactIdentity)
            json.attribute("source_artifact_identity",
                           *file.sourceArtifactIdentity);
        });
      }
    });
    json.attributeArray("external_files", [&] {
      for (const ResolvedExternalFile &file : specification.externalFiles) {
        json.object([&] {
          json.attribute("provider_input_slot", file.providerInputSlot);
          json.attribute("local_file_key", file.localFileKey);
          json.attribute("path", file.absolutePath);
          json.attribute("content_sha256",
                         formatExternalFileFingerprint(file.fingerprint));
        });
      }
    });
    json.attributeBegin("declared_outputs");
    writeStringArray(json, specification.declaredOutputs);
    json.attributeEnd();
    json.attribute("stdout", kStdoutPath);
    json.attribute("stderr", kStderrPath);
    json.attribute("completion_record", kCompletionPath);
  });
  output << '\n';
  return output.str().str();
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

std::string renderContainerInvocation(
    const std::vector<std::string> &arguments,
    const ExternalToolInvocationBundleSpec &specification) {
  std::string rendered =
      shellQuote(specification.runtime.polyArchContainer->executable);
  rendered += " 'run' '--os' " + shellQuote(*specification.runtime.os);
  rendered += " '--workdir' \"$loom_bundle_root\" '--env' 'INHERIT' '--'";
  for (const std::string &argument : arguments)
    rendered += " " + shellQuote(argument);
  return rendered;
}

std::string
renderCommand(const std::vector<std::string> &command,
              const ExternalToolInvocationBundleSpec &specification) {
  if (specification.runtime.kind == InvocationRuntimeKind::PolyArchContainer) {
    std::vector<std::string> containerArguments{
        "/usr/bin/bash", "-c", "cd -- \"$HOME/work\" || exit 126\nexec \"$@\"",
        "loom-container-entry"};
    containerArguments.insert(containerArguments.end(), command.begin(),
                              command.end());
    return renderContainerInvocation(containerArguments, specification);
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

std::vector<std::string>
frozenModules(const ExternalToolInvocationBundleSpec &specification) {
  std::vector<std::string> modules;
  auto append = [&](const ResolvedToolBinding &binding) {
    for (const std::string &module : binding.loadedModules)
      if (std::find(modules.begin(), modules.end(), module) == modules.end())
        modules.push_back(module);
  };
  if (specification.runtime.polyArchContainer)
    append(*specification.runtime.polyArchContainer);
  append(specification.tool);
  return modules;
}

std::optional<std::string>
moduleInitializer(const ExternalToolInvocationBundleSpec &specification) {
  if (specification.tool.moduleInit)
    return specification.tool.moduleInit;
  if (specification.runtime.polyArchContainer)
    return specification.runtime.polyArchContainer->moduleInit;
  return std::nullopt;
}

void appendFailure(std::string &script, llvm::StringRef status, int code) {
  script += "  loom_publish_completion " + shellQuote(status) + " " +
            std::to_string(code) + "\n";
  script += "  exit " + std::to_string(code) + "\n";
}

void appendContentDigestCheck(std::string &script, llvm::StringRef path,
                              llvm::StringRef expectedDigest) {
  script += "loom_digest=''\n";
  script += "if ! IFS= read -r -N 64 loom_digest < <(sha256sum --zero -- " +
            shellQuote(path) +
            " 2>/dev/null) || "
            "[[ \"$loom_digest\" != " +
            shellQuote(expectedDigest) + " ]]; then\n";
  appendFailure(script, "bundle_content_mismatch", 121);
  script += "fi\n";
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
  appendFailure(script, "version_mismatch", 123);
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
    appendFailure(script, "version_mismatch", 123);
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
  appendFailure(script, "version_mismatch", 123);
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

void appendContainerToolVersionCheck(
    std::string &script,
    const ExternalToolInvocationBundleSpec &specification) {
  std::vector<std::string> arguments{
      "/usr/bin/bash",
      "-c",
      "cd -- \"$HOME/work\" || exit 126\n"
      "loom_version_path=$1\n"
      "shift\n"
      "\"$@\" >\"$loom_version_path\" 2>&1",
      "loom-container-version",
      kToolVersionPath.str(),
      specification.tool.executable,
  };
  arguments.insert(arguments.end(),
                   specification.toolVersionProbe.arguments.begin(),
                   specification.toolVersionProbe.arguments.end());
  script += "loom_tool_version_file=" + shellQuote(kToolVersionPath) + "\n";
  script += "rm -f -- \"$loom_tool_version_file\"\n";
  script += renderContainerInvocation(arguments, specification) +
            " >/dev/null 2>&1\n";
  script += "loom_status=$?\n";
  appendVersionStatusCheck(script, specification.toolVersionProbe);
  script += "if [[ ! -f \"$loom_tool_version_file\" ]]; then\n";
  appendFailure(script, "version_mismatch", 123);
  script += "fi\n";
  script += "loom_version_output=$(<\"$loom_tool_version_file\")\n";
  script += "rm -f -- \"$loom_tool_version_file\"\n";
  script += "loom_tool_version_file=''\n";
  appendVersionOutputCheck(script, specification.toolVersionProbe,
                           specification.tool.version);
}

std::string
renderRunScript(const ExternalToolInvocationBundleSpec &specification) {
  std::string script =
      "#!/usr/bin/env bash\n"
      "set -u\n"
      "loom_bundle_root=$(CDPATH= cd -- \"$(dirname -- "
      "\"${BASH_SOURCE[0]}\")\" && pwd -P)\n"
      "cd -- \"$loom_bundle_root\" || exit 126\n"
      "loom_completion=" +
      shellQuote(kCompletionPath) +
      "\n"
      "loom_completion_partial=\"${loom_completion}.partial.$$\"\n"
      "loom_tool_version_file=''\n"
      "rm -f -- \"$loom_completion\" \"$loom_completion_partial\"\n"
      "trap 'rm -f -- \"$loom_completion_partial\"; if [[ -n "
      "\"$loom_tool_version_file\" ]]; then rm -f -- "
      "\"$loom_tool_version_file\"; fi' EXIT\n"
      "loom_publish_completion() {\n"
      "  printf "
      "'{\"schema\":\"loom.external_tool_completion\",\"version\":\"1.0\","
      "\"status\":\"%s\",\"exit_code\":%s}\\n' \"$1\" \"$2\" "
      ">\"$loom_completion_partial\" || exit 126\n"
      "  mv -f -- \"$loom_completion_partial\" \"$loom_completion\" || exit "
      "126\n"
      "}\n";

  script += "if ! command -v sha256sum >/dev/null 2>&1; then\n";
  appendFailure(script, "bundle_content_mismatch", 121);
  script += "fi\n";
  for (const MaterializedBundleFile &file : specification.files)
    appendContentDigestCheck(script, file.relativePath,
                             contentDigest(file.contents));
  for (const ResolvedExternalFile &file : specification.externalFiles)
    appendContentDigestCheck(
        script, file.absolutePath,
        formatExternalFileFingerprint(file.fingerprint));

  for (const std::string &name : specification.inheritEnvironment) {
    script += "if [[ -z \"${" + name + "+x}\" ]]; then\n";
    appendFailure(script, "missing_environment", 125);
    script += "fi\n";
  }

  const std::vector<std::string> modules = frozenModules(specification);
  if (!modules.empty()) {
    if (std::optional<std::string> init = moduleInitializer(specification)) {
      script += "if [[ ! -r " + shellQuote(*init) + " ]] || ! source " +
                shellQuote(*init) + " >/dev/null 2>&1; then\n";
      appendFailure(script, "module_activation_failed", 124);
      script += "fi\n";
    }
    script += "if ! type module >/dev/null 2>&1; then\n";
    appendFailure(script, "module_activation_failed", 124);
    script += "fi\n";
    for (const std::string &module : modules) {
      script +=
          "if ! module load " + shellQuote(module) + " >/dev/null 2>&1; then\n";
      appendFailure(script, "module_activation_failed", 124);
      script += "fi\n";
    }
  }

  script +=
      "if [[ ! -x " + shellQuote(specification.tool.executable) + " ]]; then\n";
  appendFailure(script, "version_mismatch", 123);
  script += "fi\n";
  if (specification.runtime.polyArchContainer) {
    script += "if [[ ! -x " +
              shellQuote(specification.runtime.polyArchContainer->executable) +
              " ]]; then\n";
    appendFailure(script, "version_mismatch", 123);
    script += "fi\n";
    appendVersionCheck(
        script,
        renderDirectCommand(specification.runtime.polyArchContainer->executable,
                            specification.containerVersionProbe),
        specification.containerVersionProbe,
        specification.runtime.polyArchContainer->version);
  }
  if (specification.runtime.kind == InvocationRuntimeKind::PolyArchContainer)
    appendContainerToolVersionCheck(script, specification);
  else
    appendVersionCheck(script,
                       renderDirectCommand(specification.tool.executable,
                                           specification.toolVersionProbe),
                       specification.toolVersionProbe,
                       specification.tool.version);

  script += "loom_status=0\n";
  script += "{\n";
  for (const std::vector<std::string> &command : specification.commands) {
    script += "  if (( loom_status == 0 )); then\n";
    script +=
        "    " + renderCommand(command, specification) + " || loom_status=$?\n";
    script += "  fi\n";
  }
  script +=
      "} >" + shellQuote(kStdoutPath) + " 2>" + shellQuote(kStderrPath) + "\n";
  script += "if (( loom_status != 0 )); then\n";
  script += "  loom_publish_completion 'tool_exit' \"$loom_status\"\n";
  script += "  exit \"$loom_status\"\n";
  script += "fi\n";
  for (const std::string &output : specification.declaredOutputs) {
    script += "if [[ ! -e " + shellQuote(output) + " ]]; then\n";
    appendFailure(script, "missing_output", 122);
    script += "fi\n";
  }
  script += "loom_publish_completion 'success' 0\n";
  return script;
}

llvm::Error writeFile(const std::filesystem::path &path, llvm::StringRef data,
                      bool executable) {
  std::error_code directoryError;
  std::filesystem::create_directories(path.parent_path(), directoryError);
  if (directoryError)
    return bundleError("could not create bundle directory: " +
                       directoryError.message());
  std::error_code outputError;
  llvm::raw_fd_ostream output(path.string(), outputError,
                              llvm::sys::fs::OF_None);
  if (outputError)
    return bundleError("could not open bundle file: " + outputError.message());
  output.write(data.data(), data.size());
  output.close();
  if (output.has_error())
    return bundleError("could not write bundle file");
  if (executable) {
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 std::filesystem::perm_options::replace,
                                 outputError);
    if (outputError)
      return bundleError("could not set bundle file permissions: " +
                         outputError.message());
  }
  return llvm::Error::success();
}

llvm::Expected<std::filesystem::path>
createStagingDirectory(const std::filesystem::path &bundleRoot) {
  for (unsigned attempt = 0; attempt != 32; ++attempt) {
    llvm::SmallString<256> model(
        (bundleRoot.string() + ".partial-%%%%%%").c_str());
    llvm::SmallString<256> candidate;
    llvm::sys::fs::createUniquePath(model, candidate, true);
    std::error_code error;
    if (std::filesystem::create_directory(candidate.str().str(), error))
      return std::filesystem::path(candidate.str().str());
    if (error != std::errc::file_exists)
      return bundleError("could not create bundle staging directory: " +
                         error.message());
  }
  return bundleError("could not allocate a bundle staging directory");
}

struct StagingCleanup {
  std::filesystem::path path;
  bool published = false;

  ~StagingCleanup() {
    if (published)
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

} // namespace

llvm::Error finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification) {
  if (llvm::Error error = validateSpecification(specification))
    return error;
  if (bundleRoot.empty() || containsNull(bundleRoot) ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return bundleError("bundle root must be an absolute path");
  const std::filesystem::path root(bundleRoot.str());
  std::error_code statusError;
  if (std::filesystem::exists(root, statusError) || statusError)
    return bundleError("bundle root already exists or is inaccessible");
  if (!std::filesystem::is_directory(root.parent_path(), statusError) ||
      statusError)
    return bundleError("bundle parent must be an existing directory");

  llvm::Expected<std::filesystem::path> staging = createStagingDirectory(root);
  if (!staging)
    return staging.takeError();
  StagingCleanup cleanup{*staging};
  std::error_code directoryError;
  std::filesystem::create_directories(*staging / "drivers", directoryError);
  std::filesystem::create_directories(*staging / "inputs", directoryError);
  std::filesystem::create_directories(*staging / "outputs", directoryError);
  if (directoryError)
    return bundleError("could not create bundle layout: " +
                       directoryError.message());
  for (const std::string &output : specification.declaredOutputs) {
    std::filesystem::create_directories((*staging / output).parent_path(),
                                        directoryError);
    if (directoryError)
      return bundleError("could not create declared output directory: " +
                         directoryError.message());
  }

  for (const MaterializedBundleFile &file : specification.files)
    if (llvm::Error error = writeFile(*staging / file.relativePath,
                                      file.contents, file.executable))
      return error;
  if (llvm::Error error = writeFile(*staging / kManifestName.str(),
                                    serializeManifest(specification), false))
    return error;
  if (llvm::Error error = writeFile(*staging / kRunScriptName.str(),
                                    renderRunScript(specification), true))
    return error;

  std::error_code publishError;
  std::filesystem::rename(*staging, root, publishError);
  if (publishError)
    return bundleError("could not publish bundle: " + publishError.message());
  cleanup.published = true;
  return llvm::Error::success();
}

llvm::Expected<int>
executeExternalToolInvocationBundle(llvm::StringRef bundleRoot) {
  if (bundleRoot.empty() || containsNull(bundleRoot) ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return bundleError("bundle root must be an absolute path");
  llvm::SmallString<256> script(bundleRoot);
  llvm::sys::path::append(script, kRunScriptName);
  llvm::ErrorOr<std::string> bash = llvm::sys::findProgramByName("bash");
  if (!bash)
    return bundleError("could not find bash: " + bash.getError().message());
  const llvm::SmallVector<llvm::StringRef, 2> arguments{*bash, script};
  std::string message;
  bool executionFailed = false;
  const int status = llvm::sys::ExecuteAndWait(
      *bash, arguments, std::nullopt, {}, 0, 0, &message, &executionFailed);
  if (executionFailed || status < 0)
    return bundleError("could not execute generated run script: " + message);
  return status;
}

llvm::Expected<InvocationCompletion>
loadExternalToolInvocationCompletion(llvm::StringRef bundleRoot) {
  if (bundleRoot.empty() || containsNull(bundleRoot) ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return bundleError("bundle root must be an absolute path");
  llvm::SmallString<256> path(bundleRoot);
  llvm::sys::path::append(path, kCompletionPath);
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return bundleError("completion record is missing or unreadable");
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return bundleError("completion record is malformed");
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return bundleError("completion record has an invalid shape");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> status = object->getString("status");
  const std::optional<std::int64_t> exitCode = object->getInteger("exit_code");
  if (!schema || *schema != "loom.external_tool_completion" || !version ||
      *version != "1.0" || !status || !exitCode || *exitCode < 0 ||
      *exitCode > 255)
    return bundleError("completion record fields are invalid");
  std::optional<InvocationCompletionStatus> parsedStatus =
      parseCompletionStatus(*status);
  if (!parsedStatus ||
      ((*parsedStatus == InvocationCompletionStatus::Success) !=
       (*exitCode == 0)))
    return bundleError("completion status and exit code are inconsistent");
  const std::string canonical =
      "{\"schema\":\"loom.external_tool_completion\",\"version\":\"1.0\","
      "\"status\":\"" +
      status->str() + "\",\"exit_code\":" + std::to_string(*exitCode) + "}\n";
  if ((*buffer)->getBuffer() != canonical)
    return bundleError("completion record is not canonical");
  return InvocationCompletion{*parsedStatus, static_cast<int>(*exitCode)};
}

} // namespace loom::external_tool

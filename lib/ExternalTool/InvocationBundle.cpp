#include "ExternalTool/InvocationBundle.h"

#include "InvocationBundleInternal.h"

#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::external_tool {

char IncompleteExternalToolInvocationError::ID = 0;

llvm::StringRef completionStatusSpelling(InvocationCompletionStatus status) {
  switch (status) {
  case InvocationCompletionStatus::Success:
    return "success";
  case InvocationCompletionStatus::MissingEnvironment:
    return "missing_environment";
  case InvocationCompletionStatus::ModuleActivationFailed:
    return "module_activation_failed";
  case InvocationCompletionStatus::VersionMismatch:
    return "version_mismatch";
  case InvocationCompletionStatus::BundleContentMismatch:
    return "bundle_content_mismatch";
  case InvocationCompletionStatus::ToolExit:
    return "tool_exit";
  case InvocationCompletionStatus::MissingOutput:
    return "missing_output";
  }
  llvm_unreachable("closed invocation completion status");
}

namespace {

llvm::Error bundleError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invocation_bundle_invalid: " + message);
}

bool containsNull(llvm::StringRef value) {
  return value.find('\0') != llvm::StringRef::npos;
}

/// The single path-domain predicate for bundle roots, shared by finalization
/// and by every open of the published directory.
llvm::Error validateBundleRootSpelling(llvm::StringRef bundleRoot) {
  if (bundleRoot.empty() || containsNull(bundleRoot) ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return bundleError("bundle root must be an absolute path");
  const std::filesystem::path root(bundleRoot.str());
  if (root.lexically_normal() != root)
    return bundleError("bundle root must be lexically normalized");
  return llvm::Error::success();
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

llvm::StringRef runtimeKindName(InvocationRuntimeKind kind) {
  switch (kind) {
  case InvocationRuntimeKind::Host:
    return "host";
  case InvocationRuntimeKind::PolyArchContainer:
    return "polyarch_container";
  }
  llvm_unreachable("unknown invocation runtime kind");
}

llvm::Expected<InvocationRuntimeKind>
parseRuntimeKind(llvm::StringRef spelling) {
  if (spelling == runtimeKindName(InvocationRuntimeKind::Host))
    return InvocationRuntimeKind::Host;
  if (spelling == runtimeKindName(InvocationRuntimeKind::PolyArchContainer))
    return InvocationRuntimeKind::PolyArchContainer;
  return bundleError("runtime binding kind is unknown");
}

std::optional<InvocationCompletionStatus>
parseCompletionStatus(llvm::StringRef spelling) {
  constexpr std::array statuses{
      InvocationCompletionStatus::Success,
      InvocationCompletionStatus::MissingEnvironment,
      InvocationCompletionStatus::ModuleActivationFailed,
      InvocationCompletionStatus::VersionMismatch,
      InvocationCompletionStatus::BundleContentMismatch,
      InvocationCompletionStatus::ToolExit,
      InvocationCompletionStatus::MissingOutput};
  for (InvocationCompletionStatus status : statuses)
    if (spelling == completionStatusSpelling(status))
      return status;
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
  if (specification.semanticContract.providerIdentity.empty() ||
      containsNull(specification.semanticContract.providerIdentity))
    return bundleError("provider identity is empty or contains NUL");
  auto importerIdentity =
      parseBlobDigestHex(specification.semanticContract.resultImporterIdentity);
  if (!importerIdentity)
    return bundleError("result importer identity is invalid: " +
                       llvm::toString(importerIdentity.takeError()));
  if (const auto *closure = std::get_if<CandidateGeneratorInvocationClosure>(
          &specification.semanticContract.semanticClosure)) {
    if (closure->typedInputBindings.empty() || closure->resolvedBinding.empty())
      return bundleError(
          "candidate generator closure carries empty owner bytes");
  } else {
    const ArtifactRootReference &request = std::get<ArtifactRootReference>(
        specification.semanticContract.semanticClosure);
    if (request.schemaIdentity.empty() || containsNull(request.schemaIdentity))
      return bundleError("evaluation closure has an invalid request reference");
  }
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
  std::set<std::string> producedExecutables;
  std::string previousProducedExecutable;
  for (const std::string &spelling : specification.toolProducedExecutables) {
    auto path = normalizedRelativePath(spelling, "tool-produced executable");
    if (!path)
      return path.takeError();
    if (!isWithin(*path, "work"))
      return bundleError(
          "tool-produced executables must be strictly below work");
    if (!previousProducedExecutable.empty() &&
        previousProducedExecutable >= *path)
      return bundleError(
          "tool-produced executable paths are not canonical sorted-unique");
    previousProducedExecutable = *path;
    producedExecutables.insert(std::move(*path));
  }
  bool hasPrecedingToolCommand = false;
  std::set<std::string> referencedProducedExecutables;
  for (const std::vector<std::string> &command : specification.commands) {
    if (command.empty())
      return bundleError("invocation command is empty");
    if (command.front() == specification.tool.executable) {
      hasPrecedingToolCommand = true;
    } else if (producedExecutables.count(command.front())) {
      if (!hasPrecedingToolCommand)
        return bundleError(
            "tool-produced executable has no preceding frozen-tool command");
      referencedProducedExecutables.insert(command.front());
      for (llvm::StringRef argument : llvm::drop_begin(command))
        if (producedExecutables.count(argument.str()))
          referencedProducedExecutables.insert(argument.str());
    } else {
      return bundleError(
          "each command must begin with the frozen tool executable or one "
          "manifest-listed tool-produced executable");
    }
    for (const std::string &argument : command)
      if (containsNull(argument))
        return bundleError("command argument contains NUL");
  }
  if (referencedProducedExecutables != producedExecutables)
    return bundleError(
        "every tool-produced executable must be used directly or named by a "
        "generated controller command");

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
  for (const ResolvedExternalFileTree &tree : specification.externalFileTrees) {
    if (tree.providerInputSlot.empty() ||
        containsNull(tree.providerInputSlot) ||
        !externalSlots.insert(tree.providerInputSlot).second)
      return bundleError(
          "external file tree provider input slot is invalid or duplicated");
    if (tree.localFileTreeKey.empty() || containsNull(tree.localFileTreeKey))
      return bundleError("external file tree local key is invalid");
    if (tree.absolutePath.empty() || containsNull(tree.absolutePath))
      return bundleError("external file tree path is empty or contains NUL");
    const std::filesystem::path path(tree.absolutePath);
    if (!path.is_absolute() || path.lexically_normal() != path)
      return bundleError(
          "external file tree path must be an absolute canonical path");
    if (tree.members.empty())
      return bundleError("external file tree has no members");
    std::string previous;
    for (const ExternalFileTreeMember &member : tree.members) {
      auto relative = normalizedRelativePath(member.relativePath,
                                             "external file tree member");
      if (!relative)
        return relative.takeError();
      if (!previous.empty() && previous >= *relative)
        return bundleError(
            "external file tree members are not canonical sorted-unique");
      previous = *relative;
    }
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
    if (file.sourceArtifact &&
        (file.sourceArtifact->schemaIdentity.empty() ||
         containsNull(file.sourceArtifact->schemaIdentity)))
      return bundleError("source Artifact reference is invalid");
    if (isInput && !file.sourceArtifact)
      return bundleError(
          "materialized input lacks a source Artifact reference");
    if (isDriver && file.sourceArtifact)
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
  for (const std::string &path : producedExecutables)
    if (llvm::Error error = reservePath(paths, path))
      return error;
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

std::string formatCanonicalHex(llvm::ArrayRef<std::uint8_t> bytes) {
  return llvm::toHex(bytes, /*LowerCase=*/true);
}

llvm::Expected<std::vector<std::uint8_t>>
parseCanonicalHex(llvm::StringRef spelling, llvm::StringRef context) {
  if (spelling.size() % 2 != 0)
    return bundleError(context + " hex encoding has an odd length");
  if (!llvm::all_of(spelling, [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
      }))
    return bundleError(context + " hex encoding is not lowercase canonical");
  std::string decoded;
  if (!llvm::tryGetFromHex(spelling, decoded))
    return bundleError(context + " hex encoding is invalid");
  return std::vector<std::uint8_t>(decoded.begin(), decoded.end());
}

void writeArtifactReference(llvm::json::OStream &json,
                            const ArtifactRootReference &reference) {
  json.object([&] {
    json.attribute("schema", reference.schemaIdentity);
    json.attribute("schema_version",
                   formatSchemaVersion(reference.schemaVersion));
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
  });
}

InvocationManifestData
makeManifest(const ExternalToolInvocationBundleSpec &specification) {
  InvocationManifestData manifest{specification.semanticContract,
                                  specification.tool,
                                  specification.toolVersionProbe,
                                  specification.runtime,
                                  specification.containerVersionProbe,
                                  specification.commands,
                                  specification.inheritEnvironment,
                                  {},
                                  specification.externalFiles,
                                  specification.externalFileTrees,
                                  specification.declaredOutputs,
                                  specification.toolProducedExecutables};
  manifest.materializedFiles.reserve(specification.files.size());
  for (const MaterializedBundleFile &file : specification.files)
    manifest.materializedFiles.push_back(ManifestMaterializedFile{
        file.relativePath, file.executable, contentDigest(file.contents),
        file.sourceArtifact});
  return manifest;
}

ExternalToolInvocationBundleSpec
makeValidationSpecification(const InvocationManifestData &manifest) {
  ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = manifest.semanticContract;
  specification.tool = manifest.tool;
  specification.toolVersionProbe = manifest.toolVersionProbe;
  specification.runtime = manifest.runtime;
  specification.containerVersionProbe = manifest.containerVersionProbe;
  specification.commands = manifest.commands;
  specification.inheritEnvironment = manifest.inheritEnvironment;
  specification.declaredOutputs = manifest.declaredOutputs;
  specification.externalFiles = manifest.externalFiles;
  specification.externalFileTrees = manifest.externalFileTrees;
  specification.toolProducedExecutables = manifest.toolProducedExecutables;
  specification.files.reserve(manifest.materializedFiles.size());
  for (const ManifestMaterializedFile &file : manifest.materializedFiles)
    specification.files.push_back(MaterializedBundleFile{
        file.relativePath, {}, file.sourceArtifact, file.executable});
  return specification;
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(key)))
      return bundleError(context + " contains unknown field '" +
                         llvm::StringRef(key) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return bundleError(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return bundleError(context + " requires array field '" + key + "'");
  return value;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return bundleError(context + " requires object field '" + key + "'");
  return value;
}

llvm::Expected<bool> requireBoolean(const llvm::json::Object &object,
                                    llvm::StringRef key,
                                    llvm::StringRef context) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value)
    return bundleError(context + " requires boolean field '" + key + "'");
  return *value;
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Array &array, llvm::StringRef context) {
  std::vector<std::string> result;
  result.reserve(array.size());
  for (const llvm::json::Value &value : array) {
    std::optional<llvm::StringRef> string = value.getAsString();
    if (!string)
      return bundleError(context + " entries must be strings");
    result.push_back(string->str());
  }
  return result;
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Object &object, llvm::StringRef key,
                 llvm::StringRef context) {
  auto array = requireArray(object, key, context);
  if (!array)
    return array.takeError();
  return parseStringArray(**array, (context + "." + key).str());
}

llvm::Expected<ToolBindingSource> parseBindingSource(llvm::StringRef spelling) {
  if (spelling == "explicit")
    return ToolBindingSource::Explicit;
  if (spelling == "environment_path")
    return ToolBindingSource::EnvironmentPath;
  if (spelling == "environment_root")
    return ToolBindingSource::EnvironmentRoot;
  if (spelling == "module")
    return ToolBindingSource::Module;
  return bundleError("tool binding source is unknown");
}

llvm::Expected<ResolvedToolBinding>
parseBinding(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"tool_key", "source", "executable", "version", "requested_modules",
           "loaded_modules", "module_init", "environment_variable"}))
    return std::move(error);
  auto toolKey = requireString(object, "tool_key", context);
  if (!toolKey)
    return toolKey.takeError();
  auto sourceText = requireString(object, "source", context);
  if (!sourceText)
    return sourceText.takeError();
  auto source = parseBindingSource(*sourceText);
  if (!source)
    return source.takeError();
  auto executable = requireString(object, "executable", context);
  if (!executable)
    return executable.takeError();
  auto version = requireString(object, "version", context);
  if (!version)
    return version.takeError();
  auto requested = parseStringArray(object, "requested_modules", context);
  if (!requested)
    return requested.takeError();
  auto loaded = parseStringArray(object, "loaded_modules", context);
  if (!loaded)
    return loaded.takeError();

  std::optional<std::string> moduleInit;
  if (const llvm::json::Value *value = object.get("module_init")) {
    std::optional<llvm::StringRef> string = value->getAsString();
    if (!string)
      return bundleError(context + ".module_init must be a string");
    moduleInit = string->str();
  }
  std::optional<std::string> environmentVariable;
  if (const llvm::json::Value *value = object.get("environment_variable")) {
    std::optional<llvm::StringRef> string = value->getAsString();
    if (!string)
      return bundleError(context + ".environment_variable must be a string");
    environmentVariable = string->str();
  }
  return ResolvedToolBinding{
      toolKey->str(),        *source,
      executable->str(),     version->str(),
      std::move(*requested), std::move(*loaded),
      std::move(moduleInit), std::move(environmentVariable)};
}

llvm::Expected<ToolVersionProbe>
parseVersionProbe(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"arguments", "required_output_substring", "accepted_exit_codes",
           "selected_output_line_substring"}))
    return std::move(error);
  auto arguments = parseStringArray(object, "arguments", context);
  if (!arguments)
    return arguments.takeError();
  auto exitCodes = requireArray(object, "accepted_exit_codes", context);
  if (!exitCodes)
    return exitCodes.takeError();
  std::vector<int> parsedExitCodes;
  parsedExitCodes.reserve((*exitCodes)->size());
  for (const llvm::json::Value &value : **exitCodes) {
    std::optional<std::uint64_t> code = value.getAsUINT64();
    if (!code || *code > 255)
      return bundleError(context +
                         ".accepted_exit_codes entries must be uint8 values");
    parsedExitCodes.push_back(static_cast<int>(*code));
  }
  auto parseOptionalString =
      [&](llvm::StringRef key) -> llvm::Expected<std::optional<std::string>> {
    const llvm::json::Value *value = object.get(key);
    if (!value)
      return std::optional<std::string>{};
    std::optional<llvm::StringRef> string = value->getAsString();
    if (!string)
      return bundleError(context + "." + key + " must be a string");
    return std::optional<std::string>(string->str());
  };
  auto marker = parseOptionalString("required_output_substring");
  if (!marker)
    return marker.takeError();
  auto selector = parseOptionalString("selected_output_line_substring");
  if (!selector)
    return selector.takeError();
  return ToolVersionProbe{std::move(*arguments), std::move(*marker),
                          std::move(parsedExitCodes), std::move(*selector)};
}

llvm::Expected<ArtifactRootReference>
parseArtifactReference(const llvm::json::Object &object,
                       llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "schema_version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  if (!schema)
    return schema.takeError();
  auto versionText = requireString(object, "schema_version", context);
  if (!versionText)
    return versionText.takeError();
  auto artifactText = requireString(object, "artifact", context);
  if (!artifactText)
    return artifactText.takeError();
  auto version = parseSchemaVersion(*versionText);
  if (!version)
    return version.takeError();
  auto artifact = parseArtifactIdentityHex(*artifactText);
  if (!artifact)
    return artifact.takeError();
  return ArtifactRootReference{schema->str(), *version, std::move(*artifact)};
}

llvm::Expected<InvocationRuntimeBinding>
parseRuntimeBinding(const llvm::json::Object &object,
                    ToolVersionProbe &containerVersionProbe) {
  auto kind = requireString(object, "kind", "runtime binding");
  if (!kind)
    return kind.takeError();
  auto rejected =
      parseStringArray(object, "rejected_compositions", "runtime binding");
  if (!rejected)
    return rejected.takeError();
  auto parsedKind = parseRuntimeKind(*kind);
  if (!parsedKind)
    return parsedKind.takeError();
  if (*parsedKind == InvocationRuntimeKind::Host) {
    if (llvm::Error error = rejectUnknownFields(
            object, "runtime binding", {"kind", "rejected_compositions"}))
      return std::move(error);
    InvocationRuntimeBinding runtime;
    runtime.kind = InvocationRuntimeKind::Host;
    runtime.rejectedCompositions = std::move(*rejected);
    return runtime;
  }
  if (llvm::Error error = rejectUnknownFields(
          object, "runtime binding",
          {"kind", "os", "container_binding", "container_version_probe",
           "rejected_compositions"}))
    return std::move(error);
  auto os = requireString(object, "os", "runtime binding");
  if (!os)
    return os.takeError();
  auto bindingObject =
      requireObject(object, "container_binding", "runtime binding");
  if (!bindingObject)
    return bindingObject.takeError();
  auto probeObject =
      requireObject(object, "container_version_probe", "runtime binding");
  if (!probeObject)
    return probeObject.takeError();
  auto binding = parseBinding(**bindingObject, "container binding");
  if (!binding)
    return binding.takeError();
  auto probe = parseVersionProbe(**probeObject, "container version probe");
  if (!probe)
    return probe.takeError();
  containerVersionProbe = std::move(*probe);
  InvocationRuntimeBinding runtime;
  runtime.kind = *parsedKind;
  runtime.polyArchContainer = std::move(*binding);
  runtime.os = os->str();
  runtime.rejectedCompositions = std::move(*rejected);
  return runtime;
}

llvm::Expected<SemanticInvocationClosure>
parseSemanticClosure(const llvm::json::Object &object,
                     llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(object, context,
                                              {"form", "typed_input_bindings",
                                               "resolved_binding",
                                               "binding_identity", "request"}))
    return std::move(error);
  auto form = requireString(object, "form", context);
  if (!form)
    return form.takeError();
  if (*form == "candidate_generator") {
    CandidateGeneratorInvocationClosure closure;
    auto inputBindings = requireString(object, "typed_input_bindings", context);
    if (!inputBindings)
      return inputBindings.takeError();
    auto inputBytes = parseCanonicalHex(*inputBindings, context);
    if (!inputBytes)
      return inputBytes.takeError();
    closure.typedInputBindings = std::move(*inputBytes);
    auto resolvedBinding = requireString(object, "resolved_binding", context);
    if (!resolvedBinding)
      return resolvedBinding.takeError();
    auto bindingBytes = parseCanonicalHex(*resolvedBinding, context);
    if (!bindingBytes)
      return bindingBytes.takeError();
    closure.resolvedBinding = std::move(*bindingBytes);
    auto identity = requireString(object, "binding_identity", context);
    if (!identity)
      return identity.takeError();
    auto identityBytes = parseCanonicalHex(*identity, context);
    if (!identityBytes)
      return identityBytes.takeError();
    auto digest = loom::BlobDigest::fromBytes(*identityBytes);
    if (!digest)
      return bundleError(context + " binding identity is not a digest");
    closure.bindingIdentity = digest->bytes();
    return SemanticInvocationClosure(std::move(closure));
  }
  if (*form == "evaluation") {
    auto request = requireObject(object, "request", context);
    if (!request)
      return request.takeError();
    auto reference = parseArtifactReference(**request, context);
    if (!reference)
      return reference.takeError();
    return SemanticInvocationClosure(std::move(*reference));
  }
  return bundleError(context + " has an unknown closure form");
}

llvm::Expected<InvocationManifestData> parseManifest(llvm::StringRef contents) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contents);
  if (!parsed)
    return bundleError("invocation manifest is malformed: " +
                       llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return bundleError("invocation manifest root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "invocation manifest",
          {"schema", "version", "provider_identity", "semantic_closure",
           "result_importer_identity", "tool_binding", "tool_version_probe",
           "runtime_binding", "commands", "inherit_environment",
           "materialized_files", "external_files", "external_file_trees",
           "tool_produced_executables", "declared_outputs", "stdout", "stderr",
           "completion_record"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "invocation manifest");
  if (!schema)
    return schema.takeError();
  auto version = requireString(*root, "version", "invocation manifest");
  if (!version)
    return version.takeError();
  if (*schema != externalToolInvocationManifestSchema)
    return bundleError("invocation manifest schema is unsupported");
  if (*version == "1.0")
    return bundleError("invocation manifest 1.0 free semantic identity is "
                       "not supported");
  if (*version != kTypedClosureManifestVersion &&
      *version != kExternalFileTreeManifestVersion &&
      *version != kToolProducedExecutableManifestVersion)
    return bundleError("invocation manifest schema or version is unsupported");
  if (*version == kTypedClosureManifestVersion &&
      root->get("external_file_trees"))
    return bundleError(
        "invocation manifest 2.0 cannot contain external file trees");
  if (*version != kToolProducedExecutableManifestVersion &&
      root->get("tool_produced_executables"))
    return bundleError("invocation manifest before 2.2 cannot contain "
                       "tool-produced executables");
  auto provider =
      requireString(*root, "provider_identity", "invocation manifest");
  if (!provider)
    return provider.takeError();
  auto closureObject =
      requireObject(*root, "semantic_closure", "invocation manifest");
  if (!closureObject)
    return closureObject.takeError();
  auto closure =
      parseSemanticClosure(**closureObject, "invocation manifest closure");
  if (!closure)
    return closure.takeError();
  auto importer =
      requireString(*root, "result_importer_identity", "invocation manifest");
  if (!importer)
    return importer.takeError();
  auto toolObject = requireObject(*root, "tool_binding", "invocation manifest");
  if (!toolObject)
    return toolObject.takeError();
  auto toolProbeObject =
      requireObject(*root, "tool_version_probe", "invocation manifest");
  if (!toolProbeObject)
    return toolProbeObject.takeError();
  auto runtimeObject =
      requireObject(*root, "runtime_binding", "invocation manifest");
  if (!runtimeObject)
    return runtimeObject.takeError();
  auto commandArray = requireArray(*root, "commands", "invocation manifest");
  if (!commandArray)
    return commandArray.takeError();
  auto inherited =
      parseStringArray(*root, "inherit_environment", "invocation manifest");
  if (!inherited)
    return inherited.takeError();
  auto materializedArray =
      requireArray(*root, "materialized_files", "invocation manifest");
  if (!materializedArray)
    return materializedArray.takeError();
  auto externalArray =
      requireArray(*root, "external_files", "invocation manifest");
  if (!externalArray)
    return externalArray.takeError();
  const llvm::json::Array *externalTreeArray = nullptr;
  if (*version == kExternalFileTreeManifestVersion ||
      *version == kToolProducedExecutableManifestVersion) {
    auto trees =
        requireArray(*root, "external_file_trees", "invocation manifest");
    if (!trees)
      return trees.takeError();
    externalTreeArray = *trees;
  }
  std::vector<std::string> toolProducedExecutables;
  if (*version == kToolProducedExecutableManifestVersion) {
    auto produced = parseStringArray(*root, "tool_produced_executables",
                                     "invocation manifest");
    if (!produced)
      return produced.takeError();
    toolProducedExecutables = std::move(*produced);
  }
  auto declared =
      parseStringArray(*root, "declared_outputs", "invocation manifest");
  if (!declared)
    return declared.takeError();
  auto stdoutPath = requireString(*root, "stdout", "invocation manifest");
  if (!stdoutPath)
    return stdoutPath.takeError();
  auto stderrPath = requireString(*root, "stderr", "invocation manifest");
  if (!stderrPath)
    return stderrPath.takeError();
  auto completionPath =
      requireString(*root, "completion_record", "invocation manifest");
  if (!completionPath)
    return completionPath.takeError();
  if (*stdoutPath != kStdoutPath || *stderrPath != kStderrPath ||
      *completionPath != kCompletionPath)
    return bundleError("invocation manifest internal paths are invalid");

  auto tool = parseBinding(**toolObject, "tool binding");
  if (!tool)
    return tool.takeError();
  auto toolProbe = parseVersionProbe(**toolProbeObject, "tool version probe");
  if (!toolProbe)
    return toolProbe.takeError();
  ToolVersionProbe containerProbe;
  auto runtime = parseRuntimeBinding(**runtimeObject, containerProbe);
  if (!runtime)
    return runtime.takeError();

  std::vector<std::vector<std::string>> commands;
  commands.reserve((*commandArray)->size());
  for (const llvm::json::Value &value : **commandArray) {
    const llvm::json::Array *command = value.getAsArray();
    if (!command)
      return bundleError("invocation manifest commands must be arrays");
    auto arguments = parseStringArray(*command, "invocation command");
    if (!arguments)
      return arguments.takeError();
    commands.push_back(std::move(*arguments));
  }

  std::vector<ManifestMaterializedFile> materializedFiles;
  materializedFiles.reserve((*materializedArray)->size());
  for (const llvm::json::Value &value : **materializedArray) {
    const llvm::json::Object *file = value.getAsObject();
    if (!file)
      return bundleError("materialized file record must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *file, "materialized file",
            {"path", "executable", "content_sha256", "source_artifact_ref"}))
      return std::move(error);
    auto path = requireString(*file, "path", "materialized file");
    if (!path)
      return path.takeError();
    auto executable = requireBoolean(*file, "executable", "materialized file");
    if (!executable)
      return executable.takeError();
    auto digestText =
        requireString(*file, "content_sha256", "materialized file");
    if (!digestText)
      return digestText.takeError();
    auto digest = parseBlobDigestHex(*digestText);
    if (!digest)
      return digest.takeError();
    std::optional<ArtifactRootReference> sourceArtifact;
    if (const llvm::json::Value *source = file->get("source_artifact_ref")) {
      const llvm::json::Object *reference = source->getAsObject();
      if (!reference)
        return bundleError(
            "materialized file source_artifact_ref must be an object");
      auto parsedReference =
          parseArtifactReference(*reference, "source Artifact reference");
      if (!parsedReference)
        return parsedReference.takeError();
      sourceArtifact = std::move(*parsedReference);
    }
    materializedFiles.push_back(
        ManifestMaterializedFile{path->str(), *executable, std::move(*digest),
                                 std::move(sourceArtifact)});
  }

  std::vector<ResolvedExternalFile> externalFiles;
  externalFiles.reserve((*externalArray)->size());
  for (const llvm::json::Value &value : **externalArray) {
    const llvm::json::Object *file = value.getAsObject();
    if (!file)
      return bundleError("external file record must be an object");
    if (llvm::Error error =
            rejectUnknownFields(*file, "external file",
                                {"provider_input_slot", "local_file_key",
                                 "path", "content_sha256"}))
      return std::move(error);
    auto slot = requireString(*file, "provider_input_slot", "external file");
    if (!slot)
      return slot.takeError();
    auto key = requireString(*file, "local_file_key", "external file");
    if (!key)
      return key.takeError();
    auto path = requireString(*file, "path", "external file");
    if (!path)
      return path.takeError();
    auto digestText = requireString(*file, "content_sha256", "external file");
    if (!digestText)
      return digestText.takeError();
    auto digest = parseExternalFileFingerprint(*digestText);
    if (!digest)
      return digest.takeError();
    externalFiles.push_back(ResolvedExternalFile{
        slot->str(), key->str(), path->str(), std::move(*digest)});
  }

  std::vector<ResolvedExternalFileTree> externalFileTrees;
  if (externalTreeArray)
    for (const llvm::json::Value &value : *externalTreeArray) {
      const llvm::json::Object *tree = value.getAsObject();
      if (!tree)
        return bundleError("external file tree record must be an object");
      if (llvm::Error error =
              rejectUnknownFields(*tree, "external file tree",
                                  {"provider_input_slot", "local_file_tree_key",
                                   "path", "members"}))
        return std::move(error);
      auto slot =
          requireString(*tree, "provider_input_slot", "external file tree");
      if (!slot)
        return slot.takeError();
      auto key =
          requireString(*tree, "local_file_tree_key", "external file tree");
      if (!key)
        return key.takeError();
      auto path = requireString(*tree, "path", "external file tree");
      if (!path)
        return path.takeError();
      auto memberArray = requireArray(*tree, "members", "external file tree");
      if (!memberArray)
        return memberArray.takeError();
      std::vector<ExternalFileTreeMember> members;
      members.reserve((*memberArray)->size());
      for (const llvm::json::Value &memberValue : **memberArray) {
        const llvm::json::Object *member = memberValue.getAsObject();
        if (!member)
          return bundleError("external file tree member must be an object");
        if (llvm::Error error =
                rejectUnknownFields(*member, "external file tree member",
                                    {"path", "content_sha256"}))
          return std::move(error);
        auto memberPath =
            requireString(*member, "path", "external file tree member");
        if (!memberPath)
          return memberPath.takeError();
        auto digestText = requireString(*member, "content_sha256",
                                        "external file tree member");
        if (!digestText)
          return digestText.takeError();
        auto digest = parseExternalFileFingerprint(*digestText);
        if (!digest)
          return digest.takeError();
        members.push_back({memberPath->str(), std::move(*digest)});
      }
      externalFileTrees.push_back(ResolvedExternalFileTree{
          slot->str(), key->str(), path->str(), std::move(members)});
    }

  InvocationManifestData manifest{
      ExternalToolSemanticContract{provider->str(), std::move(*closure),
                                   importer->str()},
      std::move(*tool),
      std::move(*toolProbe),
      std::move(*runtime),
      std::move(containerProbe),
      std::move(commands),
      std::move(*inherited),
      std::move(materializedFiles),
      std::move(externalFiles),
      std::move(externalFileTrees),
      std::move(*declared),
      std::move(toolProducedExecutables)};
  ExternalToolInvocationBundleSpec validation =
      makeValidationSpecification(manifest);
  if (llvm::Error error = validateSpecification(validation))
    return std::move(error);
  if (contents != serializeManifest(manifest, *version))
    return bundleError("invocation manifest is not canonical");
  return manifest;
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

class BundleFileDescriptor final {
public:
  explicit BundleFileDescriptor(int value = -1) : value_(value) {}
  BundleFileDescriptor(const BundleFileDescriptor &) = delete;
  BundleFileDescriptor &operator=(const BundleFileDescriptor &) = delete;
  BundleFileDescriptor(BundleFileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  BundleFileDescriptor &operator=(BundleFileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        ::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~BundleFileDescriptor() {
    if (value_ >= 0)
      ::close(value_);
  }

  int get() const { return value_; }

private:
  int value_;
};

llvm::Error bundleSystemError(const llvm::Twine &message) {
  return bundleError(message + ": " + std::strerror(errno));
}

bool sameObservedFile(const struct stat &lhs, const struct stat &rhs) {
  return lhs.st_dev == rhs.st_dev && lhs.st_ino == rhs.st_ino &&
         lhs.st_mode == rhs.st_mode && lhs.st_nlink == rhs.st_nlink &&
         lhs.st_size == rhs.st_size &&
         lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec &&
         lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec &&
         lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec &&
         lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
}

llvm::Expected<BundleFileDescriptor>
openBundleRoot(llvm::StringRef bundleRoot) {
  if (llvm::Error error = validateBundleRootSpelling(bundleRoot))
    return std::move(error);
  BundleFileDescriptor descriptor(
      ::open(bundleRoot.str().c_str(),
             O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
  if (descriptor.get() < 0)
    return bundleSystemError("could not open bundle root");
  return descriptor;
}

llvm::Expected<BundleFileDescriptor>
openOrdinaryBundleFile(int bundleRoot, llvm::StringRef relativePath) {
  auto normalized = normalizedRelativePath(relativePath, "bundle file");
  if (!normalized)
    return normalized.takeError();

  BundleFileDescriptor current(::fcntl(bundleRoot, F_DUPFD_CLOEXEC, 0));
  if (current.get() < 0)
    return bundleSystemError("could not duplicate bundle root descriptor");

  std::vector<std::string> components;
  for (const std::filesystem::path &component :
       std::filesystem::path(*normalized))
    components.push_back(component.string());
  for (std::size_t index = 0; index < components.size(); ++index) {
    const bool final = index + 1 == components.size();
    struct stat status{};
    if (::fstatat(current.get(), components[index].c_str(), &status,
                  AT_SYMLINK_NOFOLLOW) != 0)
      return bundleSystemError("could not inspect bundle file component '" +
                               components[index] + "'");
    if (S_ISLNK(status.st_mode))
      return bundleError("bundle file path contains a symlink component");
    if (final && !S_ISREG(status.st_mode))
      return bundleError("bundle file path must name an ordinary file");
    if (!final && !S_ISDIR(status.st_mode))
      return bundleError("bundle file parent is not an ordinary directory");

    const int flags = final ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK
                            : O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY;
    BundleFileDescriptor next(
        ::openat(current.get(), components[index].c_str(), flags));
    if (next.get() < 0)
      return bundleSystemError("could not open bundle file component '" +
                               components[index] + "'");
    current = std::move(next);
  }
  return current;
}

llvm::Expected<std::string>
readOrdinaryBundleFile(int bundleRoot, llvm::StringRef relativePath) {
  auto file = openOrdinaryBundleFile(bundleRoot, relativePath);
  if (!file)
    return file.takeError();
  struct stat before{};
  if (::fstat(file->get(), &before) != 0)
    return bundleSystemError("could not inspect opened bundle file");
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return bundleError("bundle file path must name an ordinary file");
  if (static_cast<std::uintmax_t>(before.st_size) >
      std::numeric_limits<std::size_t>::max())
    return bundleError("bundle file is too large to import");

  std::string contents;
  contents.reserve(static_cast<std::size_t>(before.st_size));
  std::array<char, 64 * 1024> buffer{};
  while (true) {
    const ssize_t count = ::read(file->get(), buffer.data(), buffer.size());
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return bundleSystemError("could not read bundle file");
    }
    contents.append(buffer.data(), static_cast<std::size_t>(count));
  }

  struct stat after{};
  if (::fstat(file->get(), &after) != 0)
    return bundleSystemError("could not re-inspect opened bundle file");
  if (!sameObservedFile(before, after) ||
      contents.size() != static_cast<std::uintmax_t>(after.st_size))
    return bundleError("bundle file changed while it was read");
  return contents;
}

std::string serializeCompletion(InvocationCompletionStatus status, int exitCode,
                                const BlobDigest &manifestDigest,
                                llvm::ArrayRef<BlobDigest> outputDigests) {
  std::string canonical =
      "{\"schema\":\"loom.external_tool_completion\",\"version\":\"1.0\","
      "\"status\":\"" +
      completionStatusSpelling(status).str() +
      "\",\"exit_code\":" + std::to_string(exitCode) +
      ",\"manifest_sha256\":\"" + formatBlobDigestHex(manifestDigest) +
      "\",\"output_sha256\":[";
  for (std::size_t index = 0; index < outputDigests.size(); ++index) {
    if (index != 0)
      canonical += ',';
    canonical += "\"" + formatBlobDigestHex(outputDigests[index]) + "\"";
  }
  canonical += "]}\n";
  return canonical;
}

llvm::Expected<InvocationCompletion> parseCompletion(llvm::StringRef contents) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contents);
  if (!parsed)
    return bundleError("completion record is malformed: " +
                       llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 6)
    return bundleError("completion record has an invalid shape");
  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> status = object->getString("status");
  const std::optional<std::int64_t> exitCode = object->getInteger("exit_code");
  const std::optional<llvm::StringRef> manifestText =
      object->getString("manifest_sha256");
  const llvm::json::Array *outputArray = object->getArray("output_sha256");
  if (!schema || *schema != "loom.external_tool_completion" || !version ||
      *version != "1.0" || !status || !exitCode || *exitCode < 0 ||
      *exitCode > 255 || !manifestText || !outputArray)
    return bundleError("completion record fields are invalid");
  std::optional<InvocationCompletionStatus> parsedStatus =
      parseCompletionStatus(*status);
  if (!parsedStatus ||
      ((*parsedStatus == InvocationCompletionStatus::Success) !=
       (*exitCode == 0)))
    return bundleError("completion status and exit code are inconsistent");
  auto manifestDigest = parseBlobDigestHex(*manifestText);
  if (!manifestDigest)
    return manifestDigest.takeError();
  std::vector<BlobDigest> outputDigests;
  outputDigests.reserve(outputArray->size());
  for (const llvm::json::Value &value : *outputArray) {
    std::optional<llvm::StringRef> digestText = value.getAsString();
    if (!digestText)
      return bundleError("completion output digest must be a string");
    auto digest = parseBlobDigestHex(*digestText);
    if (!digest)
      return digest.takeError();
    outputDigests.push_back(std::move(*digest));
  }
  if (*parsedStatus != InvocationCompletionStatus::Success &&
      !outputDigests.empty())
    return bundleError("failed completion record contains output digests");
  if (contents != serializeCompletion(*parsedStatus,
                                      static_cast<int>(*exitCode),
                                      *manifestDigest, outputDigests))
    return bundleError("completion record is not canonical");
  return InvocationCompletion{*parsedStatus, static_cast<int>(*exitCode),
                              std::move(*manifestDigest),
                              std::move(outputDigests)};
}

/// Reads and parses the completion record through one already-open bundle
/// root descriptor; the single completion consumer shared by strict import
/// and the diagnostic reader.
llvm::Expected<InvocationCompletion> readCompletionFromRoot(int bundleRoot) {
  auto contents = readOrdinaryBundleFile(bundleRoot, kCompletionPath);
  if (!contents)
    return contents.takeError();
  return parseCompletion(*contents);
}

/// One owning view of a prepared bundle: the open root descriptor plus the
/// exact manifest bytes, proven to digest to the prepared handle. Strict
/// import reads the manifest, completion, and every declared output through
/// this one descriptor.
struct PreparedBundleView final {
  BundleFileDescriptor root;
  std::string manifestBytes;
};

llvm::Expected<PreparedBundleView>
openPreparedBundle(const PreparedExternalToolInvocation &prepared) {
  auto root = openBundleRoot(prepared.bundleRoot);
  if (!root)
    return root.takeError();
  auto contents = readOrdinaryBundleFile(root->get(), kManifestName);
  if (!contents)
    return contents.takeError();
  if (contentDigest(*contents) != prepared.manifestDigest)
    return bundleError(
        "invocation manifest does not match the prepared handle");
  return PreparedBundleView{std::move(*root), std::move(*contents)};
}

} // namespace

void writeToolVersionProbeJson(llvm::json::OStream &json,
                               const ToolVersionProbe &probe) {
  json.object([&] {
    json.attributeArray("arguments", [&] {
      for (const std::string &argument : probe.arguments)
        json.value(argument);
    });
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

BlobDigest contentDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

llvm::Expected<std::pair<std::string, InvocationManifestData>>
loadPreparedInvocationManifest(const PreparedExternalToolInvocation &prepared) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  auto manifest = parseManifest(bundle->manifestBytes);
  if (!manifest)
    return manifest.takeError();
  return std::make_pair(std::move(bundle->manifestBytes), std::move(*manifest));
}

std::string
serializeInvocationCompletion(InvocationCompletionStatus status, int exitCode,
                              const BlobDigest &manifestDigest,
                              llvm::ArrayRef<BlobDigest> outputDigests) {
  return serializeCompletion(status, exitCode, manifestDigest, outputDigests);
}

llvm::Expected<BlobDigest> deriveExternalToolExecutionBindingDigest(
    const ResolvedToolBinding &tool, const InvocationRuntimeBinding &runtime) {
  if (llvm::Error error = validateBinding(tool, "tool binding"))
    return std::move(error);
  if (static_cast<unsigned>(runtime.kind) >
      static_cast<unsigned>(InvocationRuntimeKind::PolyArchContainer))
    return bundleError("runtime binding kind is unknown");
  if (runtime.kind == InvocationRuntimeKind::Host) {
    if (runtime.polyArchContainer || runtime.os)
      return bundleError("host runtime may not contain a container binding");
  } else {
    if (!runtime.polyArchContainer || !runtime.os || runtime.os->empty() ||
        containsNull(*runtime.os))
      return bundleError(
          "PolyArch/container runtime requires a binding and OS");
    if (llvm::Error error =
            validateBinding(*runtime.polyArchContainer, "container binding"))
      return std::move(error);
    if (runtime.polyArchContainer->toolKey != "polyarch_container")
      return bundleError("container binding has the wrong logical tool key");
    if (tool.moduleInit && runtime.polyArchContainer->moduleInit &&
        tool.moduleInit != runtime.polyArchContainer->moduleInit)
      return bundleError(
          "tool and container bindings use different module initializers");
  }
  for (const std::string &rejection : runtime.rejectedCompositions)
    if (rejection.empty() || containsNull(rejection))
      return bundleError("runtime rejection provenance is invalid");
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 0);
  json.object([&] {
    json.attribute("schema", "loom.external_tool_execution_binding");
    json.attribute("version", "1.0");
    json.attributeBegin("tool_binding");
    writeBinding(json, tool);
    json.attributeEnd();
    json.attributeObject("runtime_binding", [&] {
      json.attribute("kind", runtimeKindName(runtime.kind));
      if (runtime.kind == InvocationRuntimeKind::PolyArchContainer) {
        json.attribute("os", *runtime.os);
        json.attributeBegin("container_binding");
        writeBinding(json, *runtime.polyArchContainer);
        json.attributeEnd();
      }
    });
  });
  return contentDigest(output.str());
}

llvm::Expected<BlobDigest> deriveExternalToolExecutionBindingDigest(
    const PreparedExternalToolInvocation &prepared) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  auto manifest = parseManifest(bundle->manifestBytes);
  if (!manifest)
    return manifest.takeError();
  return deriveExternalToolExecutionBindingDigest(manifest->tool,
                                                  manifest->runtime);
}

llvm::Expected<std::string> deriveExternalToolResultImporterIdentity(
    llvm::ArrayRef<std::uint8_t> semanticDescriptorReferenceBytes,
    ProviderForm providerForm) {
  if (semanticDescriptorReferenceBytes.empty())
    return bundleError("semantic descriptor reference is empty");
  if (providerForm != ProviderForm::ExternalPrepareImport)
    return bundleError(
        "result importer identity requires ExternalPrepareImport");

  static constexpr llvm::StringLiteral domain =
      "loom.external_tool_importer.v1";
  std::vector<std::uint8_t> preimage;
  preimage.reserve(domain.size() + 1 + 8 +
                   semanticDescriptorReferenceBytes.size() + 4);
  preimage.insert(preimage.end(), domain.bytes_begin(), domain.bytes_end());
  preimage.push_back(0);
  const std::uint64_t referenceSize = semanticDescriptorReferenceBytes.size();
  for (unsigned shift = 56; shift != 0; shift -= 8)
    preimage.push_back(static_cast<std::uint8_t>(referenceSize >> shift));
  preimage.push_back(static_cast<std::uint8_t>(referenceSize));
  preimage.insert(preimage.end(), semanticDescriptorReferenceBytes.begin(),
                  semanticDescriptorReferenceBytes.end());
  const std::uint32_t form = static_cast<std::uint32_t>(providerForm);
  preimage.push_back(static_cast<std::uint8_t>(form >> 24));
  preimage.push_back(static_cast<std::uint8_t>(form >> 16));
  preimage.push_back(static_cast<std::uint8_t>(form >> 8));
  preimage.push_back(static_cast<std::uint8_t>(form));
  return formatBlobDigestHex(computeBlobDigest(preimage));
}

std::string serializeManifest(const InvocationManifestData &manifest,
                              llvm::StringRef version) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", externalToolInvocationManifestSchema);
    json.attribute("version", version);
    json.attribute("provider_identity",
                   manifest.semanticContract.providerIdentity);
    json.attributeObject("semantic_closure", [&] {
      if (const auto *closure =
              std::get_if<CandidateGeneratorInvocationClosure>(
                  &manifest.semanticContract.semanticClosure)) {
        json.attribute("form", "candidate_generator");
        json.attribute("typed_input_bindings",
                       formatCanonicalHex(closure->typedInputBindings));
        json.attribute("resolved_binding",
                       formatCanonicalHex(closure->resolvedBinding));
        json.attribute("binding_identity",
                       formatCanonicalHex(closure->bindingIdentity));
      } else {
        json.attribute("form", "evaluation");
        json.attributeBegin("request");
        writeArtifactReference(json,
                               std::get<ArtifactRootReference>(
                                   manifest.semanticContract.semanticClosure));
        json.attributeEnd();
      }
    });
    json.attribute("result_importer_identity",
                   manifest.semanticContract.resultImporterIdentity);
    json.attributeBegin("tool_binding");
    writeBinding(json, manifest.tool);
    json.attributeEnd();
    json.attributeBegin("tool_version_probe");
    writeToolVersionProbeJson(json, manifest.toolVersionProbe);
    json.attributeEnd();
    json.attributeObject("runtime_binding", [&] {
      json.attribute("kind", runtimeKindName(manifest.runtime.kind));
      if (manifest.runtime.kind == InvocationRuntimeKind::PolyArchContainer) {
        json.attribute("os", *manifest.runtime.os);
        json.attributeBegin("container_binding");
        writeBinding(json, *manifest.runtime.polyArchContainer);
        json.attributeEnd();
        json.attributeBegin("container_version_probe");
        writeToolVersionProbeJson(json, manifest.containerVersionProbe);
        json.attributeEnd();
      }
      json.attributeBegin("rejected_compositions");
      writeStringArray(json, manifest.runtime.rejectedCompositions);
      json.attributeEnd();
    });
    json.attributeArray("commands", [&] {
      for (const std::vector<std::string> &command : manifest.commands)
        writeStringArray(json, command);
    });
    if (version == kToolProducedExecutableManifestVersion) {
      json.attributeBegin("tool_produced_executables");
      writeStringArray(json, manifest.toolProducedExecutables);
      json.attributeEnd();
    }
    json.attributeBegin("inherit_environment");
    writeStringArray(json, manifest.inheritEnvironment);
    json.attributeEnd();
    json.attributeArray("materialized_files", [&] {
      for (const ManifestMaterializedFile &file : manifest.materializedFiles) {
        json.object([&] {
          json.attribute("path", file.relativePath);
          json.attribute("executable", file.executable);
          json.attribute("content_sha256",
                         formatBlobDigestHex(file.contentDigest));
          if (file.sourceArtifact) {
            json.attributeBegin("source_artifact_ref");
            writeArtifactReference(json, *file.sourceArtifact);
            json.attributeEnd();
          }
        });
      }
    });
    json.attributeArray("external_files", [&] {
      for (const ResolvedExternalFile &file : manifest.externalFiles) {
        json.object([&] {
          json.attribute("provider_input_slot", file.providerInputSlot);
          json.attribute("local_file_key", file.localFileKey);
          json.attribute("path", file.absolutePath);
          json.attribute("content_sha256",
                         formatExternalFileFingerprint(file.fingerprint));
        });
      }
    });
    if (version == kExternalFileTreeManifestVersion ||
        version == kToolProducedExecutableManifestVersion)
      json.attributeArray("external_file_trees", [&] {
        for (const ResolvedExternalFileTree &tree :
             manifest.externalFileTrees) {
          json.object([&] {
            json.attribute("provider_input_slot", tree.providerInputSlot);
            json.attribute("local_file_tree_key", tree.localFileTreeKey);
            json.attribute("path", tree.absolutePath);
            json.attributeArray("members", [&] {
              for (const ExternalFileTreeMember &member : tree.members) {
                json.object([&] {
                  json.attribute("path", member.relativePath);
                  json.attribute(
                      "content_sha256",
                      formatExternalFileFingerprint(member.fingerprint));
                });
              }
            });
          });
        }
      });
    json.attributeBegin("declared_outputs");
    writeStringArray(json, manifest.declaredOutputs);
    json.attributeEnd();
    json.attribute("stdout", kStdoutPath);
    json.attribute("stderr", kStderrPath);
    json.attribute("completion_record", kCompletionPath);
  });
  output << '\n';
  return output.str().str();
}

llvm::Expected<PreparedExternalToolInvocation>
finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification) {
  std::uint64_t previousDiagnosticOrdinal = 0;
  bool hasPreviousDiagnosticOrdinal = false;
  for (const std::uint64_t ordinal : specification.diagnosticCommandOrdinals) {
    if (ordinal >= specification.commands.size())
      return bundleError("diagnostic command ordinal is out of range");
    if (hasPreviousDiagnosticOrdinal && previousDiagnosticOrdinal >= ordinal)
      return bundleError(
          "diagnostic command ordinals are not canonical sorted-unique");
    previousDiagnosticOrdinal = ordinal;
    hasPreviousDiagnosticOrdinal = true;
    if (specification.commands[ordinal].empty() ||
        !llvm::is_contained(specification.toolProducedExecutables,
                            specification.commands[ordinal].front()))
      return bundleError(
          "diagnostic command is not a tool-produced executable");
  }
  for (const std::vector<std::string> &command : specification.commands)
    for (const std::string &argument : command)
      if (isDiagnosticVerbosityBinding(argument))
        return bundleError(
            "diagnostic verbosity is owned by invocation finalization");

  if (llvm::Error error = validateSpecification(specification))
    return error;
  if (llvm::Error error = validateBundleRootSpelling(bundleRoot))
    return std::move(error);
  const std::filesystem::path root(bundleRoot.str());
  std::error_code statusError;
  if (std::filesystem::exists(root, statusError) || statusError)
    return bundleError("bundle root already exists or is inaccessible");
  if (!std::filesystem::is_directory(root.parent_path(), statusError) ||
      statusError)
    return bundleError("bundle parent must be an existing directory");
  InvocationManifestData manifest = makeManifest(specification);
  if (std::optional<std::string> argument = diagnosticVerbosityArgument())
    for (const std::uint64_t ordinal : specification.diagnosticCommandOrdinals)
      manifest.commands[ordinal].push_back(*argument);

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
  for (const std::string &executable : specification.toolProducedExecutables) {
    std::filesystem::create_directories((*staging / executable).parent_path(),
                                        directoryError);
    if (directoryError)
      return bundleError(
          "could not create tool-produced executable directory: " +
          directoryError.message());
  }

  for (const MaterializedBundleFile &file : specification.files)
    if (llvm::Error error = writeFile(*staging / file.relativePath,
                                      file.contents, file.executable))
      return error;
  const std::string manifestBytes = serializeManifest(manifest);
  if (llvm::Error error =
          writeFile(*staging / kManifestName.str(), manifestBytes, false))
    return error;
  if (llvm::Error error = writeFile(*staging / kRunScriptName.str(),
                                    renderRunScript(manifest), true))
    return error;

  std::error_code publishError;
  std::filesystem::rename(*staging, root, publishError);
  if (publishError)
    return bundleError("could not publish bundle: " + publishError.message());
  cleanup.published = true;
  return PreparedExternalToolInvocation{bundleRoot.str(),
                                        contentDigest(manifestBytes)};
}

llvm::Expected<InvocationCompletion> loadExternalToolInvocationCompletion(
    const PreparedExternalToolInvocation &prepared) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  return readCompletionFromRoot(bundle->root.get());
}

llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  const int bundleRoot = bundle->root.get();
  auto manifest = parseManifest(bundle->manifestBytes);
  if (!manifest)
    return manifest.takeError();
  if (manifest->semanticContract.providerIdentity !=
      expectation.semanticContract.providerIdentity)
    return bundleError("invocation provider identity does not match importer");
  if (manifest->semanticContract.semanticClosure !=
      expectation.semanticContract.semanticClosure)
    return bundleError("invocation semantic closure does not match importer");
  if (manifest->semanticContract.resultImporterIdentity !=
      expectation.semanticContract.resultImporterIdentity)
    return bundleError("invocation result importer identity does not match");

  std::vector<ExternalToolInvocationSemanticInput> semanticInputs;
  for (const ManifestMaterializedFile &file : manifest->materializedFiles)
    if (file.sourceArtifact)
      semanticInputs.push_back(ExternalToolInvocationSemanticInput{
          file.relativePath, *file.sourceArtifact, file.contentDigest});
  if (semanticInputs != expectation.semanticInputs)
    return bundleError("invocation semantic inputs do not match importer");
  std::vector<ExternalToolInvocationExternalInput> externalInputs;
  externalInputs.reserve(manifest->externalFiles.size());
  for (const ResolvedExternalFile &file : manifest->externalFiles)
    externalInputs.push_back(ExternalToolInvocationExternalInput{
        file.providerInputSlot, file.fingerprint});
  if (externalInputs != expectation.externalInputs)
    return bundleError("invocation external inputs do not match importer");
  std::vector<ExternalToolInvocationExternalFileTree> externalFileTrees;
  externalFileTrees.reserve(manifest->externalFileTrees.size());
  for (const ResolvedExternalFileTree &tree : manifest->externalFileTrees)
    externalFileTrees.push_back(ExternalToolInvocationExternalFileTree{
        tree.providerInputSlot, tree.members});
  if (externalFileTrees != expectation.externalFileTrees)
    return bundleError("invocation external file trees do not match importer");
  if (manifest->declaredOutputs != expectation.declaredOutputs)
    return bundleError("invocation declared outputs do not match importer");

  // Only an absent completion record is the typed incomplete-attempt signal;
  // every present-but-unreadable or malformed record stays an ordinary
  // integrity failure through the shared reader below.
  struct stat completionStatus{};
  if (::fstatat(bundleRoot, kCompletionPath.str().c_str(), &completionStatus,
                AT_SYMLINK_NOFOLLOW) != 0 &&
      errno == ENOENT)
    return ExternalToolInvocationAttemptOutcome(
        IncompleteExternalToolInvocationAttempt{});
  auto completion = readCompletionFromRoot(bundleRoot);
  if (!completion)
    return completion.takeError();
  if (completion->manifestDigest != contentDigest(bundle->manifestBytes))
    return bundleError("completion does not bind the imported manifest");
  if (completion->status != InvocationCompletionStatus::Success)
    return ExternalToolInvocationAttemptOutcome(
        FailedExternalToolInvocationAttempt{completion->status,
                                            completion->exitCode});
  if (completion->outputDigests.size() != manifest->declaredOutputs.size())
    return bundleError("completion output digest count is invalid");

  std::vector<std::pair<std::string, std::string>> outputs;
  outputs.reserve(manifest->declaredOutputs.size());
  for (std::size_t index = 0; index < manifest->declaredOutputs.size();
       ++index) {
    const std::string &path = manifest->declaredOutputs[index];
    auto output = readOrdinaryBundleFile(bundleRoot, path);
    if (!output)
      return output.takeError();
    if (contentDigest(*output) != completion->outputDigests[index])
      return bundleError("declared output does not match completion digest");
    outputs.emplace_back(path, std::move(*output));
  }
  return ExternalToolInvocationAttemptOutcome(
      ImportedExternalToolInvocationBundle(std::move(outputs)));
}

llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation) {
  auto attempt = importExternalToolInvocationAttempt(prepared, expectation);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (std::holds_alternative<FailedExternalToolInvocationAttempt>(*attempt))
    return bundleError("invocation did not complete successfully");
  return std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
}

llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
    const ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath) {
  auto normalized = normalizedRelativePath(relativePath, "declared output");
  if (!normalized)
    return normalized.takeError();
  const auto found = llvm::find_if(bundle.outputs_, [&](const auto &output) {
    return output.first == *normalized;
  });
  if (found == bundle.outputs_.end())
    return bundleError("output is not declared by the invocation manifest");
  return found->second;
}

} // namespace loom::external_tool

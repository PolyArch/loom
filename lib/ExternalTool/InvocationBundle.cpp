#include "ExternalTool/InvocationBundle.h"

#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
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
namespace {

constexpr llvm::StringLiteral kManifestName = "tool-invocation.json";
constexpr llvm::StringLiteral kRunScriptName = "run.sh";
constexpr llvm::StringLiteral kCompletionPath = "outputs/completion.json";
constexpr llvm::StringLiteral kExecutionStartPath =
    "outputs/.loom-execution-started";
constexpr llvm::StringLiteral kStdoutPath = "outputs/stdout.log";
constexpr llvm::StringLiteral kStderrPath = "outputs/stderr.log";
constexpr llvm::StringLiteral kToolVersionPath = "outputs/.loom-tool-version";

struct ManifestMaterializedFile final {
  std::string relativePath;
  bool executable;
  BlobDigest contentDigest;
  std::optional<ArtifactRootReference> sourceArtifact;
};

struct InvocationManifestData final {
  std::string providerIdentity;
  std::string semanticBindingIdentity;
  std::string resultImporterIdentity;
  ResolvedToolBinding tool;
  ToolVersionProbe toolVersionProbe;
  InvocationRuntimeBinding runtime;
  ToolVersionProbe containerVersionProbe;
  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> inheritEnvironment;
  std::vector<ManifestMaterializedFile> materializedFiles;
  std::vector<ResolvedExternalFile> externalFiles;
  std::vector<std::string> declaredOutputs;
};

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
                              kCompletionPath.str(), kExecutionStartPath.str(),
                              kStdoutPath.str(),     kStderrPath.str(),
                              kToolVersionPath.str()};
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

BlobDigest contentDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
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
  InvocationManifestData manifest{specification.providerIdentity,
                                  specification.semanticBindingIdentity,
                                  specification.resultImporterIdentity,
                                  specification.tool,
                                  specification.toolVersionProbe,
                                  specification.runtime,
                                  specification.containerVersionProbe,
                                  specification.commands,
                                  specification.inheritEnvironment,
                                  {},
                                  specification.externalFiles,
                                  specification.declaredOutputs};
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
  specification.providerIdentity = manifest.providerIdentity;
  specification.semanticBindingIdentity = manifest.semanticBindingIdentity;
  specification.resultImporterIdentity = manifest.resultImporterIdentity;
  specification.tool = manifest.tool;
  specification.toolVersionProbe = manifest.toolVersionProbe;
  specification.runtime = manifest.runtime;
  specification.containerVersionProbe = manifest.containerVersionProbe;
  specification.commands = manifest.commands;
  specification.inheritEnvironment = manifest.inheritEnvironment;
  specification.declaredOutputs = manifest.declaredOutputs;
  specification.externalFiles = manifest.externalFiles;
  specification.files.reserve(manifest.materializedFiles.size());
  for (const ManifestMaterializedFile &file : manifest.materializedFiles)
    specification.files.push_back(MaterializedBundleFile{
        file.relativePath, {}, file.sourceArtifact, file.executable});
  return specification;
}

std::string serializeManifest(const InvocationManifestData &manifest) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", "loom.external_tool_invocation");
    json.attribute("version", "1.0");
    json.attribute("provider_identity", manifest.providerIdentity);
    json.attribute("semantic_binding_identity",
                   manifest.semanticBindingIdentity);
    json.attribute("result_importer_identity", manifest.resultImporterIdentity);
    json.attributeBegin("tool_binding");
    writeBinding(json, manifest.tool);
    json.attributeEnd();
    json.attributeBegin("tool_version_probe");
    writeVersionProbe(json, manifest.toolVersionProbe);
    json.attributeEnd();
    json.attributeObject("runtime_binding", [&] {
      if (manifest.runtime.kind == InvocationRuntimeKind::Host) {
        json.attribute("kind", "host");
      } else {
        json.attribute("kind", "polyarch_container");
        json.attribute("os", *manifest.runtime.os);
        json.attributeBegin("container_binding");
        writeBinding(json, *manifest.runtime.polyArchContainer);
        json.attributeEnd();
        json.attributeBegin("container_version_probe");
        writeVersionProbe(json, manifest.containerVersionProbe);
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
  if (*kind == "host") {
    if (llvm::Error error = rejectUnknownFields(
            object, "runtime binding", {"kind", "rejected_compositions"}))
      return std::move(error);
    InvocationRuntimeBinding runtime;
    runtime.kind = InvocationRuntimeKind::Host;
    runtime.rejectedCompositions = std::move(*rejected);
    return runtime;
  }
  if (*kind != "polyarch_container")
    return bundleError("runtime binding kind is unknown");
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
  runtime.kind = InvocationRuntimeKind::PolyArchContainer;
  runtime.polyArchContainer = std::move(*binding);
  runtime.os = os->str();
  runtime.rejectedCompositions = std::move(*rejected);
  return runtime;
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
          {"schema", "version", "provider_identity",
           "semantic_binding_identity", "result_importer_identity",
           "tool_binding", "tool_version_probe", "runtime_binding", "commands",
           "inherit_environment", "materialized_files", "external_files",
           "declared_outputs", "stdout", "stderr", "completion_record"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "invocation manifest");
  if (!schema)
    return schema.takeError();
  auto version = requireString(*root, "version", "invocation manifest");
  if (!version)
    return version.takeError();
  if (*schema != "loom.external_tool_invocation" || *version != "1.0")
    return bundleError("invocation manifest schema or version is unsupported");
  auto provider =
      requireString(*root, "provider_identity", "invocation manifest");
  if (!provider)
    return provider.takeError();
  auto semantic =
      requireString(*root, "semantic_binding_identity", "invocation manifest");
  if (!semantic)
    return semantic.takeError();
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

  InvocationManifestData manifest{
      provider->str(),           semantic->str(),
      importer->str(),           std::move(*tool),
      std::move(*toolProbe),     std::move(*runtime),
      std::move(containerProbe), std::move(commands),
      std::move(*inherited),     std::move(materializedFiles),
      std::move(externalFiles),  std::move(*declared)};
  ExternalToolInvocationBundleSpec validation =
      makeValidationSpecification(manifest);
  if (llvm::Error error = validateSpecification(validation))
    return std::move(error);
  if (contents != serializeManifest(manifest))
    return bundleError("invocation manifest is not canonical");
  return manifest;
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

std::string renderContainerInvocation(const std::vector<std::string> &arguments,
                                      const InvocationManifestData &manifest) {
  std::string rendered =
      shellQuote(manifest.runtime.polyArchContainer->executable);
  rendered += " 'run' '--os' " + shellQuote(*manifest.runtime.os);
  rendered += " '--workdir' \"$loom_bundle_root\" '--env' 'INHERIT' '--'";
  for (const std::string &argument : arguments)
    rendered += " " + shellQuote(argument);
  return rendered;
}

std::string renderCommand(const std::vector<std::string> &command,
                          const InvocationManifestData &manifest) {
  if (manifest.runtime.kind == InvocationRuntimeKind::PolyArchContainer) {
    std::vector<std::string> containerArguments{
        "/usr/bin/bash", "-c", "cd -- \"$HOME/work\" || exit 126\nexec \"$@\"",
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

void appendContainerToolVersionCheck(std::string &script,
                                     const InvocationManifestData &manifest) {
  std::vector<std::string> arguments{
      "/usr/bin/bash",
      "-c",
      "cd -- \"$HOME/work\" || exit 126\n"
      "loom_version_path=$1\n"
      "shift\n"
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
  appendFailure(script, "version_mismatch", 123);
  script += "fi\n";
  script += "loom_version_output=$(<\"$loom_tool_version_file\")\n";
  script += "rm -f -- \"$loom_tool_version_file\"\n";
  script += "loom_tool_version_file=''\n";
  appendVersionOutputCheck(script, manifest.toolVersionProbe,
                           manifest.tool.version);
}

std::string renderRunScript(const InvocationManifestData &manifest) {
  const std::string manifestDigest =
      formatBlobDigestHex(contentDigest(serializeManifest(manifest)));
  std::string script =
      "#!/usr/bin/env bash\n"
      "set -u\n"
      "loom_bundle_root=$(CDPATH= cd -- \"$(dirname -- "
      "\"${BASH_SOURCE[0]}\")\" && pwd -P)\n"
      "cd -- \"$loom_bundle_root\" || exit 126\n"
      "loom_execution_start=" +
      shellQuote(kExecutionStartPath) +
      "\n"
      "if ! mkdir -- \"$loom_execution_start\"; then exit 120; fi\n"
      "loom_completion=" +
      shellQuote(kCompletionPath) +
      "\n"
      "loom_completion_partial=\"${loom_completion}.partial.$$\"\n"
      "loom_tool_version_file=''\n"
      "loom_manifest_digest=" +
      shellQuote(manifestDigest) +
      "\n"
      "loom_output_digests='[]'\n"
      "if [[ -e \"$loom_completion\" || -L \"$loom_completion\" ]]; then "
      "exit 120; fi\n"
      "rm -f -- \"$loom_completion_partial\"\n"
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
      ">\"$loom_completion_partial\" || exit 126\n"
      "  mv -f -- \"$loom_completion_partial\" \"$loom_completion\" || exit "
      "126\n"
      "}\n";

  script += "if ! command -v sha256sum >/dev/null 2>&1; then\n";
  appendFailure(script, "bundle_content_mismatch", 121);
  script += "fi\n";
  appendContentDigestCheck(script, kManifestName, manifestDigest);
  for (const ManifestMaterializedFile &file : manifest.materializedFiles)
    appendContentDigestCheck(script, file.relativePath,
                             formatBlobDigestHex(file.contentDigest));
  for (const ResolvedExternalFile &file : manifest.externalFiles)
    appendContentDigestCheck(script, file.absolutePath,
                             formatExternalFileFingerprint(file.fingerprint));
  for (const std::string &output : manifest.declaredOutputs) {
    script += "if [[ -e " + shellQuote(output) + " || -L " +
              shellQuote(output) + " ]]; then\n";
    appendFailure(script, "bundle_content_mismatch", 121);
    script += "fi\n";
  }

  for (const std::string &name : manifest.inheritEnvironment) {
    script += "if [[ -z \"${" + name + "+x}\" ]]; then\n";
    appendFailure(script, "missing_environment", 125);
    script += "fi\n";
  }

  const std::vector<std::string> modules = frozenModules(manifest);
  if (!modules.empty()) {
    if (std::optional<std::string> init = moduleInitializer(manifest)) {
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
      "if [[ ! -x " + shellQuote(manifest.tool.executable) + " ]]; then\n";
  appendFailure(script, "version_mismatch", 123);
  script += "fi\n";
  if (manifest.runtime.polyArchContainer) {
    script += "if [[ ! -x " +
              shellQuote(manifest.runtime.polyArchContainer->executable) +
              " ]]; then\n";
    appendFailure(script, "version_mismatch", 123);
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

  script += "loom_status=0\n";
  script += "{\n";
  for (const std::vector<std::string> &command : manifest.commands) {
    script += "  if (( loom_status == 0 )); then\n";
    script +=
        "    " + renderCommand(command, manifest) + " || loom_status=$?\n";
    script += "  fi\n";
  }
  script +=
      "} >" + shellQuote(kStdoutPath) + " 2>" + shellQuote(kStderrPath) + "\n";
  script += "if (( loom_status != 0 )); then\n";
  script += "  loom_publish_completion 'tool_exit' \"$loom_status\"\n";
  script += "  exit \"$loom_status\"\n";
  script += "fi\n";
  script += "loom_success_output_digests='['\n";
  for (std::size_t index = 0; index < manifest.declaredOutputs.size();
       ++index) {
    const std::string &output = manifest.declaredOutputs[index];
    script += "if [[ ! -f " + shellQuote(output) + " || -L " +
              shellQuote(output) + " ]]; then\n";
    appendFailure(script, "missing_output", 122);
    script += "fi\n";
    script += "loom_output_digest=''\n";
    script +=
        "if ! IFS= read -r -N 64 loom_output_digest < <(sha256sum --zero -- " +
        shellQuote(output) + " 2>/dev/null); then\n";
    appendFailure(script, "missing_output", 122);
    script += "fi\n";
    if (index != 0)
      script += "loom_success_output_digests+=','\n";
    script +=
        "loom_success_output_digests+=\"\\\"${loom_output_digest}\\\"\"\n";
  }
  script += "loom_success_output_digests+=']'\n";
  script += "loom_output_digests=\"$loom_success_output_digests\"\n";
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
  if (bundleRoot.empty() || containsNull(bundleRoot) ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return bundleError("bundle root must be an absolute path");
  const std::filesystem::path root(bundleRoot.str());
  if (root.lexically_normal() != root)
    return bundleError("bundle root must be lexically normalized");
  BundleFileDescriptor descriptor(
      ::open(root.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
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

llvm::Expected<std::string>
readOrdinaryBundleFile(llvm::StringRef bundleRoot,
                       llvm::StringRef relativePath) {
  auto root = openBundleRoot(bundleRoot);
  if (!root)
    return root.takeError();
  return readOrdinaryBundleFile(root->get(), relativePath);
}

std::string serializeCompletion(llvm::StringRef status, int exitCode,
                                const BlobDigest &manifestDigest,
                                llvm::ArrayRef<BlobDigest> outputDigests) {
  std::string canonical =
      "{\"schema\":\"loom.external_tool_completion\",\"version\":\"1.0\","
      "\"status\":\"" +
      status.str() + "\",\"exit_code\":" + std::to_string(exitCode) +
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
  if (contents != serializeCompletion(*status, static_cast<int>(*exitCode),
                                      *manifestDigest, outputDigests))
    return bundleError("completion record is not canonical");
  return InvocationCompletion{*parsedStatus, static_cast<int>(*exitCode),
                              std::move(*manifestDigest),
                              std::move(outputDigests)};
}

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
  const InvocationManifestData manifest = makeManifest(specification);

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
                                    serializeManifest(manifest), false))
    return error;
  if (llvm::Error error = writeFile(*staging / kRunScriptName.str(),
                                    renderRunScript(manifest), true))
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
  auto contents = readOrdinaryBundleFile(bundleRoot, kCompletionPath);
  if (!contents)
    return contents.takeError();
  return parseCompletion(*contents);
}

llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationImportExpectation &expectation) {
  auto root = openBundleRoot(bundleRoot);
  if (!root)
    return root.takeError();
  auto contents = readOrdinaryBundleFile(root->get(), kManifestName);
  if (!contents)
    return contents.takeError();
  auto manifest = parseManifest(*contents);
  if (!manifest)
    return manifest.takeError();
  if (manifest->providerIdentity != expectation.providerIdentity)
    return bundleError("invocation provider identity does not match importer");
  if (manifest->semanticBindingIdentity != expectation.semanticBindingIdentity)
    return bundleError(
        "invocation semantic binding identity does not match importer");
  if (manifest->resultImporterIdentity != expectation.resultImporterIdentity)
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
  if (manifest->declaredOutputs != expectation.declaredOutputs)
    return bundleError("invocation declared outputs do not match importer");

  auto completionContents =
      readOrdinaryBundleFile(root->get(), kCompletionPath);
  if (!completionContents)
    return completionContents.takeError();
  auto completion = parseCompletion(*completionContents);
  if (!completion)
    return completion.takeError();
  if (completion->status != InvocationCompletionStatus::Success)
    return bundleError("invocation did not complete successfully");
  if (completion->manifestDigest != contentDigest(*contents))
    return bundleError("completion does not bind the imported manifest");
  if (completion->outputDigests.size() != manifest->declaredOutputs.size())
    return bundleError("completion output digest count is invalid");

  std::vector<std::pair<std::string, std::string>> outputs;
  outputs.reserve(manifest->declaredOutputs.size());
  for (std::size_t index = 0; index < manifest->declaredOutputs.size();
       ++index) {
    const std::string &path = manifest->declaredOutputs[index];
    auto output = readOrdinaryBundleFile(root->get(), path);
    if (!output)
      return output.takeError();
    if (contentDigest(*output) != completion->outputDigests[index])
      return bundleError("declared output does not match completion digest");
    outputs.emplace_back(path, std::move(*output));
  }
  return ImportedExternalToolInvocationBundle(std::move(outputs));
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

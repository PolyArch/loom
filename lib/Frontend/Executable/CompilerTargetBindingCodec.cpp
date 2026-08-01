#include "CompilerTargetBindingInternal.h"

#include "Common/ArtifactText.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::detail {
namespace {

llvm::Error codecError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "compiler_target_binding_invalid: " + message);
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(entry.first)))
      return codecError(context + " contains unknown field '" +
                        llvm::StringRef(entry.first) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return codecError(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return codecError(context + " requires object field '" + key + "'");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return codecError(context + " requires array field '" + key + "'");
  return value;
}

llvm::StringRef abiSpelling(fabric::RiscVAbi value) {
  switch (value) {
  case fabric::RiscVAbi::Ilp32:
    return "ilp32";
  case fabric::RiscVAbi::Ilp32e:
    return "ilp32e";
  case fabric::RiscVAbi::Ilp32f:
    return "ilp32f";
  case fabric::RiscVAbi::Ilp32d:
    return "ilp32d";
  case fabric::RiscVAbi::Lp64:
    return "lp64";
  case fabric::RiscVAbi::Lp64f:
    return "lp64f";
  case fabric::RiscVAbi::Lp64d:
    return "lp64d";
  }
  llvm_unreachable("unknown RISC-V ABI");
}

llvm::Expected<fabric::RiscVAbi> parseAbi(llvm::StringRef spelling) {
  static constexpr std::pair<llvm::StringLiteral, fabric::RiscVAbi> values[] = {
      {"ilp32", fabric::RiscVAbi::Ilp32},
      {"ilp32e", fabric::RiscVAbi::Ilp32e},
      {"ilp32f", fabric::RiscVAbi::Ilp32f},
      {"ilp32d", fabric::RiscVAbi::Ilp32d},
      {"lp64", fabric::RiscVAbi::Lp64},
      {"lp64f", fabric::RiscVAbi::Lp64f},
      {"lp64d", fabric::RiscVAbi::Lp64d},
  };
  for (const auto &[name, value] : values)
    if (name == spelling)
      return value;
  return codecError("unknown backend_abi '" + spelling + "'");
}

llvm::StringRef codeModelSpelling(fabric::RiscVCodeModel value) {
  switch (value) {
  case fabric::RiscVCodeModel::MediumLow:
    return "medium_low";
  case fabric::RiscVCodeModel::MediumAny:
    return "medium_any";
  }
  llvm_unreachable("unknown RISC-V code model");
}

llvm::Expected<fabric::RiscVCodeModel>
parseCodeModel(llvm::StringRef spelling) {
  if (spelling == "medium_low")
    return fabric::RiscVCodeModel::MediumLow;
  if (spelling == "medium_any")
    return fabric::RiscVCodeModel::MediumAny;
  return codecError("unknown code_model '" + spelling + "'");
}

llvm::StringRef relocationSpelling(fabric::RelocationModel value) {
  switch (value) {
  case fabric::RelocationModel::Static:
    return "static";
  case fabric::RelocationModel::PositionIndependent:
    return "position_independent";
  }
  llvm_unreachable("unknown relocation model");
}

llvm::Expected<fabric::RelocationModel>
parseRelocation(llvm::StringRef spelling) {
  if (spelling == "static")
    return fabric::RelocationModel::Static;
  if (spelling == "position_independent")
    return fabric::RelocationModel::PositionIndependent;
  return codecError("unknown relocation_model '" + spelling + "'");
}

llvm::StringRef scopeSpelling(fabric::InstructionSyncScope value) {
  switch (value) {
  case fabric::InstructionSyncScope::SingleThread:
    return "single_thread";
  case fabric::InstructionSyncScope::Hart:
    return "hart";
  case fabric::InstructionSyncScope::System:
    return "system";
  }
  llvm_unreachable("unknown synchronization scope");
}

llvm::Expected<fabric::InstructionSyncScope>
parseScope(llvm::StringRef spelling) {
  if (spelling == "single_thread")
    return fabric::InstructionSyncScope::SingleThread;
  if (spelling == "hart")
    return fabric::InstructionSyncScope::Hart;
  if (spelling == "system")
    return fabric::InstructionSyncScope::System;
  return codecError("unknown architecture_sync_scope_ref '" + spelling + "'");
}

llvm::StringRef supportRoleSpelling(CompilerSupportRole value) {
  switch (value) {
  case CompilerSupportRole::StartupObject:
    return "startup_object";
  case CompilerSupportRole::RuntimeLibrary:
    return "runtime_library";
  case CompilerSupportRole::BuiltinLibrary:
    return "builtin_library";
  }
  llvm_unreachable("unknown compiler support role");
}

llvm::Expected<CompilerSupportRole> parseSupportRole(llvm::StringRef spelling) {
  if (spelling == "startup_object")
    return CompilerSupportRole::StartupObject;
  if (spelling == "runtime_library")
    return CompilerSupportRole::RuntimeLibrary;
  if (spelling == "builtin_library")
    return CompilerSupportRole::BuiltinLibrary;
  return codecError("unknown support component role '" + spelling + "'");
}

llvm::StringRef linkModeSpelling(CompilerSupportLinkMode value) {
  switch (value) {
  case CompilerSupportLinkMode::Static:
    return "static";
  case CompilerSupportLinkMode::Dynamic:
    return "dynamic";
  }
  llvm_unreachable("unknown compiler support link mode");
}

llvm::Expected<CompilerSupportLinkMode>
parseLinkMode(llvm::StringRef spelling) {
  if (spelling == "static")
    return CompilerSupportLinkMode::Static;
  if (spelling == "dynamic")
    return CompilerSupportLinkMode::Dynamic;
  return codecError("unknown support component link_mode '" + spelling + "'");
}

llvm::Expected<CompilerProcessorArchitectureRef>
parseProcessorRef(const llvm::json::Object &root) {
  auto object = requireObject(root, "processor_architecture_ref",
                              "CompilerTargetBinding");
  if (!object)
    return object.takeError();
  if (llvm::Error error =
          rejectUnknownFields(**object, "processor_architecture_ref",
                              {"kind", "fabric_artifact", "local_ref"}))
    return std::move(error);
  auto kind = requireString(**object, "kind", "processor_architecture_ref");
  if (!kind)
    return kind.takeError();
  auto artifact =
      requireString(**object, "fabric_artifact", "processor_architecture_ref");
  if (!artifact)
    return artifact.takeError();
  auto identity = parseArtifactIdentityHex(*artifact);
  if (!identity)
    return identity.takeError();
  auto localRef =
      requireString(**object, "local_ref", "processor_architecture_ref");
  if (!localRef)
    return localRef.takeError();
  if (*kind == "host_core") {
    auto ref = fabric::parseFabricRef<fabric::HostCoreOccurrenceRef>(*localRef);
    if (!ref)
      return ref.takeError();
    return CompilerProcessorArchitectureRef::host({*identity, *ref});
  }
  if (*kind == "instruction_core") {
    auto ref =
        fabric::parseFabricRef<fabric::InstructionCoreContextRef>(*localRef);
    if (!ref)
      return ref.takeError();
    return CompilerProcessorArchitectureRef::instruction({*identity, *ref});
  }
  return codecError("unknown processor architecture kind '" + *kind + "'");
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Object &root, llvm::StringRef key) {
  auto array = requireArray(root, key, "CompilerTargetBinding");
  if (!array)
    return array.takeError();
  std::vector<std::string> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    std::optional<llvm::StringRef> spelling = value.getAsString();
    if (!spelling)
      return codecError(key + " entries must be strings");
    result.push_back(spelling->str());
  }
  return result;
}

llvm::Expected<std::vector<TargetScopeBinding>>
parseTargetScopes(const llvm::json::Object &root) {
  auto array =
      requireArray(root, "target_scope_bindings", "CompilerTargetBinding");
  if (!array)
    return array.takeError();
  std::vector<TargetScopeBinding> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return codecError("target_scope_bindings entries must be objects");
    if (llvm::Error error = rejectUnknownFields(
            *object, "target scope binding",
            {"architecture_sync_scope_ref", "llvm_sync_scope_id"}))
      return std::move(error);
    auto architectureScope = requireString(
        *object, "architecture_sync_scope_ref", "target scope binding");
    if (!architectureScope)
      return architectureScope.takeError();
    auto scope = parseScope(*architectureScope);
    if (!scope)
      return scope.takeError();
    auto llvmScope =
        requireString(*object, "llvm_sync_scope_id", "target scope binding");
    if (!llvmScope)
      return llvmScope.takeError();
    result.push_back({*scope, llvmScope->str()});
  }
  return result;
}

llvm::Expected<std::vector<CompilerSupportComponent>>
parseSupportComponents(const llvm::json::Object &root) {
  auto array =
      requireArray(root, "support_components", "CompilerTargetBinding");
  if (!array)
    return array.takeError();
  std::vector<CompilerSupportComponent> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return codecError("support_components entries must be objects");
    if (llvm::Error error = rejectUnknownFields(
            *object, "support component",
            {"role", "interface_abi_identity", "content_blob", "link_mode"}))
      return std::move(error);
    auto roleText = requireString(*object, "role", "support component");
    if (!roleText)
      return roleText.takeError();
    auto role = parseSupportRole(*roleText);
    if (!role)
      return role.takeError();
    auto interface =
        requireString(*object, "interface_abi_identity", "support component");
    if (!interface)
      return interface.takeError();
    auto blobText = requireString(*object, "content_blob", "support component");
    if (!blobText)
      return blobText.takeError();
    auto blob = parseBlobDigestHex(*blobText);
    if (!blob)
      return blob.takeError();
    auto modeText = requireString(*object, "link_mode", "support component");
    if (!modeText)
      return modeText.takeError();
    auto mode = parseLinkMode(*modeText);
    if (!mode)
      return mode.takeError();
    result.push_back({*role, interface->str(), *blob, *mode});
  }
  if (llvm::Error error = validateSupportComponents(result))
    return std::move(error);
  return result;
}

void writeProcessorRef(llvm::json::OStream &json,
                       const CompilerProcessorArchitectureRef &reference) {
  json.object([&] {
    std::visit(
        [&](const auto &value) {
          using Ref = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Ref,
                                       CompilerProcessorArchitectureRef::Host>)
            json.attribute("kind", "host_core");
          else
            json.attribute("kind", "instruction_core");
          json.attribute("fabric_artifact",
                         formatArtifactIdentityHex(value.artifact));
          json.attribute("local_ref", fabric::printFabricRef(value.entity));
        },
        reference.value());
  });
}

} // namespace

llvm::Error validateSupportComponents(
    llvm::ArrayRef<CompilerSupportComponent> supportComponents) {
  auto key = [](const CompilerSupportComponent &component) {
    return std::tuple{static_cast<std::uint32_t>(component.role),
                      llvm::StringRef(component.interfaceAbiIdentity),
                      component.contentBlob.bytes(),
                      static_cast<std::uint32_t>(component.linkMode)};
  };
  for (std::size_t index = 0; index < supportComponents.size(); ++index) {
    if (supportComponents[index].interfaceAbiIdentity.empty())
      return codecError("support component interface ABI identity is empty");
    if (index != 0 &&
        !(key(supportComponents[index - 1]) < key(supportComponents[index])))
      return codecError(
          "support_components are not strictly canonical and unique");
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<CompilerSupportComponent>>
canonicalizeSupportComponents(
    llvm::ArrayRef<CompilerSupportComponent> supportComponents) {
  auto key = [](const CompilerSupportComponent &component) {
    return std::tuple{static_cast<std::uint32_t>(component.role),
                      llvm::StringRef(component.interfaceAbiIdentity),
                      component.contentBlob.bytes(),
                      static_cast<std::uint32_t>(component.linkMode)};
  };
  std::vector<CompilerSupportComponent> result(supportComponents.begin(),
                                               supportComponents.end());
  llvm::sort(result, [&](const auto &lhs, const auto &rhs) {
    return key(lhs) < key(rhs);
  });
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (result[index].interfaceAbiIdentity.empty())
      return codecError("support component interface ABI identity is empty");
    if (index != 0 && key(result[index - 1]) == key(result[index]))
      return codecError("duplicate support component");
  }
  return result;
}

std::string
serializeCompilerTargetBinding(const CompilerTargetBinding &binding) {
  llvm::SmallString<2048> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", compilerTargetBindingSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(compilerTargetBindingSchema.version));
    json.attributeBegin("processor_architecture_ref");
    writeProcessorRef(json, binding.processorArchitecture());
    json.attributeEnd();
    json.attribute(
        "architecture_fingerprint",
        formatArchitectureFingerprintHex(binding.architectureFingerprint()));
    json.attributeObject("compiler_provider", [&] {
      json.attribute("repository_identity",
                     binding.compilerProvider().repositoryIdentity);
      json.attribute("full_commit_identity",
                     binding.compilerProvider().fullCommitIdentity);
    });
    json.attribute("target_triple", binding.targetTriple());
    json.attribute("data_layout", binding.dataLayout());
    json.attribute("backend_abi", abiSpelling(binding.backendAbi()));
    json.attribute("object_format", "elf");
    json.attribute("code_model", codeModelSpelling(binding.codeModel()));
    json.attribute("relocation_model",
                   relocationSpelling(binding.relocationModel()));
    json.attribute("backend_cpu", binding.backendCpu());
    json.attributeArray("backend_features", [&] {
      for (const std::string &feature : binding.backendFeatures())
        json.value(feature);
    });
    json.attributeArray("target_scope_bindings", [&] {
      for (const TargetScopeBinding &scope : binding.targetScopeBindings()) {
        json.object([&] {
          json.attribute("architecture_sync_scope_ref",
                         scopeSpelling(scope.architectureScope));
          json.attribute("llvm_sync_scope_id", scope.llvmSyncScopeId);
        });
      }
    });
    json.attributeArray("support_components", [&] {
      for (const CompilerSupportComponent &component :
           binding.supportComponents()) {
        json.object([&] {
          json.attribute("role", supportRoleSpelling(component.role));
          json.attribute("interface_abi_identity",
                         component.interfaceAbiIdentity);
          json.attribute("content_blob",
                         formatBlobDigestHex(component.contentBlob));
          json.attribute("link_mode", linkModeSpelling(component.linkMode));
        });
      }
    });
  });
  return output.str().str();
}

llvm::Expected<DecodedCompilerTargetBindingFields>
parseCompilerTargetBindingFields(llvm::StringRef jsonText) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return codecError("root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "CompilerTargetBinding",
          {"schema", "schema_version", "processor_architecture_ref",
           "architecture_fingerprint", "compiler_provider", "target_triple",
           "data_layout", "backend_abi", "object_format", "code_model",
           "relocation_model", "backend_cpu", "backend_features",
           "target_scope_bindings", "support_components"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "CompilerTargetBinding");
  if (!schema)
    return schema.takeError();
  if (*schema != compilerTargetBindingSchema.identity)
    return codecError("unsupported schema '" + *schema + "'");
  auto version =
      requireString(*root, "schema_version", "CompilerTargetBinding");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (*parsedVersion != compilerTargetBindingSchema.version)
    return codecError("unsupported schema_version '" + *version + "'");

  auto processor = parseProcessorRef(*root);
  if (!processor)
    return processor.takeError();
  auto fingerprintText =
      requireString(*root, "architecture_fingerprint", "CompilerTargetBinding");
  if (!fingerprintText)
    return fingerprintText.takeError();
  auto fingerprint = parseArchitectureFingerprintHex(*fingerprintText);
  if (!fingerprint)
    return fingerprint.takeError();
  auto providerObject =
      requireObject(*root, "compiler_provider", "CompilerTargetBinding");
  if (!providerObject)
    return providerObject.takeError();
  if (llvm::Error error =
          rejectUnknownFields(**providerObject, "compiler_provider",
                              {"repository_identity", "full_commit_identity"}))
    return std::move(error);
  auto repository = requireString(**providerObject, "repository_identity",
                                  "compiler_provider");
  if (!repository)
    return repository.takeError();
  auto commit = requireString(**providerObject, "full_commit_identity",
                              "compiler_provider");
  if (!commit)
    return commit.takeError();
  auto triple = requireString(*root, "target_triple", "CompilerTargetBinding");
  if (!triple)
    return triple.takeError();
  auto dataLayout =
      requireString(*root, "data_layout", "CompilerTargetBinding");
  if (!dataLayout)
    return dataLayout.takeError();
  auto abiText = requireString(*root, "backend_abi", "CompilerTargetBinding");
  if (!abiText)
    return abiText.takeError();
  auto abi = parseAbi(*abiText);
  if (!abi)
    return abi.takeError();
  auto objectFormat =
      requireString(*root, "object_format", "CompilerTargetBinding");
  if (!objectFormat)
    return objectFormat.takeError();
  if (*objectFormat != "elf")
    return codecError("unknown object_format '" + *objectFormat + "'");
  auto codeModelText =
      requireString(*root, "code_model", "CompilerTargetBinding");
  if (!codeModelText)
    return codeModelText.takeError();
  auto codeModel = parseCodeModel(*codeModelText);
  if (!codeModel)
    return codeModel.takeError();
  auto relocationText =
      requireString(*root, "relocation_model", "CompilerTargetBinding");
  if (!relocationText)
    return relocationText.takeError();
  auto relocation = parseRelocation(*relocationText);
  if (!relocation)
    return relocation.takeError();
  auto cpu = requireString(*root, "backend_cpu", "CompilerTargetBinding");
  if (!cpu)
    return cpu.takeError();
  auto features = parseStringArray(*root, "backend_features");
  if (!features)
    return features.takeError();
  auto scopes = parseTargetScopes(*root);
  if (!scopes)
    return scopes.takeError();
  auto support = parseSupportComponents(*root);
  if (!support)
    return support.takeError();

  return DecodedCompilerTargetBindingFields{std::move(*processor),
                                            *fingerprint,
                                            {repository->str(), commit->str()},
                                            triple->str(),
                                            dataLayout->str(),
                                            *abi,
                                            CompilerObjectFormat::Elf,
                                            *codeModel,
                                            *relocation,
                                            cpu->str(),
                                            std::move(*features),
                                            std::move(*scopes),
                                            std::move(*support)};
}

} // namespace loom::detail

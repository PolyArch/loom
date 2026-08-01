#include "InstructionCoreBinaryInternal.h"

#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::detail {
namespace {

llvm::Error codecError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(entry.first)))
      return codecError("instruction_core_binary_not_canonical",
                        context + " contains unknown field '" +
                            llvm::StringRef(entry.first) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return codecError("instruction_core_binary_invalid",
                      context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<std::uint64_t> requireU64(const llvm::json::Object &object,
                                         llvm::StringRef key,
                                         llvm::StringRef context) {
  const llvm::json::Value *raw = object.get(key);
  if (!raw)
    return codecError("instruction_core_binary_invalid",
                      context + " requires integer field '" + key + "'");
  std::optional<std::uint64_t> value = raw->getAsUINT64();
  if (!value)
    return codecError("instruction_core_binary_invalid",
                      context + " field '" + key +
                          "' must be an unsigned integer");
  return *value;
}

llvm::Expected<bool> requireBool(const llvm::json::Object &object,
                                 llvm::StringRef key, llvm::StringRef context) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value)
    return codecError("instruction_core_binary_invalid",
                      context + " requires boolean field '" + key + "'");
  return *value;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return codecError("instruction_core_binary_invalid",
                      context + " requires object field '" + key + "'");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return codecError("instruction_core_binary_invalid",
                      context + " requires array field '" + key + "'");
  return value;
}

void writeArtifactRoot(llvm::json::OStream &json,
                       const ArtifactRootReference &reference) {
  json.object([&] {
    json.attribute("schema", reference.schemaIdentity);
    json.attribute("schema_version",
                   formatSchemaVersion(reference.schemaVersion));
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
  });
}

llvm::Expected<ArtifactRootReference>
parseArtifactRoot(const llvm::json::Object &root, llvm::StringRef key) {
  auto object = requireObject(root, key, "InstructionCoreBinary");
  if (!object)
    return object.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **object, key, {"schema", "schema_version", "artifact"}))
    return std::move(error);
  auto schema = requireString(**object, "schema", key);
  if (!schema)
    return schema.takeError();
  auto version = requireString(**object, "schema_version", key);
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto artifact = requireString(**object, "artifact", key);
  if (!artifact)
    return artifact.takeError();
  auto identity = parseArtifactIdentityHex(*artifact);
  if (!identity)
    return identity.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion, *identity};
}

llvm::Expected<std::vector<InstructionLoadSegment>>
parseLoadSegments(const llvm::json::Object &root) {
  auto array = requireArray(root, "load_segments", "InstructionCoreBinary");
  if (!array)
    return array.takeError();
  std::vector<InstructionLoadSegment> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return codecError("instruction_core_binary_invalid",
                        "load_segments entries must be objects");
    if (llvm::Error error = rejectUnknownFields(
            *object, "load segment",
            {"segment_ordinal", "virtual_address", "file_offset", "file_size",
             "memory_size", "alignment", "readable", "writable", "executable"}))
      return std::move(error);
    auto ordinal = requireU64(*object, "segment_ordinal", "load segment");
    if (!ordinal)
      return ordinal.takeError();
    auto address = requireU64(*object, "virtual_address", "load segment");
    if (!address)
      return address.takeError();
    auto offset = requireU64(*object, "file_offset", "load segment");
    if (!offset)
      return offset.takeError();
    auto fileSize = requireU64(*object, "file_size", "load segment");
    if (!fileSize)
      return fileSize.takeError();
    auto memorySize = requireU64(*object, "memory_size", "load segment");
    if (!memorySize)
      return memorySize.takeError();
    auto alignment = requireU64(*object, "alignment", "load segment");
    if (!alignment)
      return alignment.takeError();
    auto readable = requireBool(*object, "readable", "load segment");
    if (!readable)
      return readable.takeError();
    auto writable = requireBool(*object, "writable", "load segment");
    if (!writable)
      return writable.takeError();
    auto executable = requireBool(*object, "executable", "load segment");
    if (!executable)
      return executable.takeError();
    if (*ordinal != result.size())
      return codecError("instruction_core_binary_not_canonical",
                        "load segment ordinals are not dense array positions");
    result.push_back({*ordinal, *address, *offset, *fileSize, *memorySize,
                      *alignment, *readable, *writable, *executable});
  }
  return result;
}

llvm::Expected<std::vector<ThreadEntryBinding>>
parseThreadEntries(const llvm::json::Object &root,
                   const ArtifactIdentity &dataflowArtifact) {
  auto array =
      requireArray(root, "thread_entry_table", "InstructionCoreBinary");
  if (!array)
    return array.takeError();
  std::vector<ThreadEntryBinding> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return codecError("instruction_core_binary_invalid",
                        "thread_entry_table entries must be objects");
    if (llvm::Error error =
            rejectUnknownFields(*object, "thread entry binding",
                                {"root_thread_launch_ref", "entry_ordinal"}))
      return std::move(error);
    auto local = requireString(*object, "root_thread_launch_ref",
                               "thread entry binding");
    if (!local)
      return local.takeError();
    auto bytes = parseArtifactLocalPayloadHex(*local);
    if (!bytes)
      return bytes.takeError();
    auto reference =
        dataflow::decodeDataflowReference<dataflow::RootThreadLaunchRef>(
            *bytes, dataflowArtifact);
    if (!reference)
      return reference.takeError();
    auto ordinal = requireU64(*object, "entry_ordinal", "thread entry binding");
    if (!ordinal)
      return ordinal.takeError();
    result.push_back({*reference, *ordinal});
  }
  return result;
}

llvm::Expected<std::vector<RuntimeImport>>
parseRuntimeImports(const llvm::json::Object &root) {
  auto array = requireArray(root, "runtime_imports", "InstructionCoreBinary");
  if (!array)
    return array.takeError();
  std::vector<RuntimeImport> result;
  result.reserve((*array)->size());
  for (const llvm::json::Value &value : **array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return codecError("instruction_core_binary_invalid",
                        "runtime_imports entries must be objects");
    if (llvm::Error error = rejectUnknownFields(
            *object, "runtime import",
            {"support_component_ordinal", "abi_symbol", "abi_symbol_version"}))
      return std::move(error);
    auto ordinal =
        requireU64(*object, "support_component_ordinal", "runtime import");
    if (!ordinal)
      return ordinal.takeError();
    auto symbol = requireString(*object, "abi_symbol", "runtime import");
    if (!symbol)
      return symbol.takeError();
    std::optional<std::string> version;
    if (const llvm::json::Value *raw = object->get("abi_symbol_version")) {
      std::optional<llvm::StringRef> spelling = raw->getAsString();
      if (!spelling)
        return codecError("instruction_core_binary_invalid",
                          "abi_symbol_version must be a string when present");
      version = spelling->str();
    }
    result.push_back({*ordinal, symbol->str(), std::move(version)});
  }
  return result;
}

} // namespace

llvm::Expected<std::vector<ThreadEntryBinding>>
canonicalizeThreadEntries(llvm::ArrayRef<ThreadEntryBinding> entries,
                          const ArtifactIdentity &dataflowArtifact) {
  std::vector<ThreadEntryBinding> result(entries.begin(), entries.end());
  for (const ThreadEntryBinding &entry : result)
    if (entry.rootThreadLaunch.artifact != dataflowArtifact)
      return codecError(
          "instruction_core_binary_foreign_root",
          "thread entry key belongs to another Dataflow artifact");
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return lhs.rootThreadLaunch.entity.value() <
           rhs.rootThreadLaunch.entity.value();
  });
  if (result.empty())
    return codecError("instruction_core_binary_missing_root",
                      "thread_entry_table must not be empty");
  for (std::size_t index = 1; index < result.size(); ++index)
    if (result[index - 1].rootThreadLaunch.entity ==
        result[index].rootThreadLaunch.entity)
      return codecError("instruction_core_binary_duplicate_root",
                        "thread_entry_table contains a duplicate root launch");
  return result;
}

llvm::Expected<std::vector<RuntimeImport>>
canonicalizeRuntimeImports(llvm::ArrayRef<RuntimeImport> imports,
                           const CompilerTargetBinding &target) {
  std::vector<RuntimeImport> result(imports.begin(), imports.end());
  for (const RuntimeImport &entry : result) {
    if (entry.supportComponentOrdinal >= target.supportComponents().size())
      return codecError("instruction_core_binary_invalid_import",
                        "runtime import support component is out of range");
    if (target.supportComponents()[entry.supportComponentOrdinal].linkMode !=
        CompilerSupportLinkMode::Dynamic)
      return codecError("instruction_core_binary_invalid_import",
                        "runtime import does not select a dynamic support "
                        "component");
    if (entry.abiSymbol.empty() ||
        entry.abiSymbol.find('\0') != std::string::npos)
      return codecError("instruction_core_binary_invalid_import",
                        "runtime import ABI symbol is empty or contains NUL");
    if (entry.abiSymbolVersion &&
        (entry.abiSymbolVersion->empty() ||
         entry.abiSymbolVersion->find('\0') != std::string::npos))
      return codecError("instruction_core_binary_invalid_import",
                        "runtime import ABI version is empty or contains NUL");
  }
  auto key = [](const RuntimeImport &entry) {
    return std::tuple{llvm::StringRef(entry.abiSymbol), entry.abiSymbolVersion,
                      entry.supportComponentOrdinal};
  };
  llvm::sort(result, [&](const auto &lhs, const auto &rhs) {
    return key(lhs) < key(rhs);
  });
  for (std::size_t index = 1; index < result.size(); ++index)
    if (result[index - 1].abiSymbol == result[index].abiSymbol &&
        result[index - 1].abiSymbolVersion == result[index].abiSymbolVersion)
      return codecError(
          "instruction_core_binary_duplicate_import",
          "one ABI import resolves to multiple support components");
  return result;
}

std::string
serializeInstructionCoreBinary(const InstructionCoreBinary &binary) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", instructionCoreBinarySchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(instructionCoreBinarySchema.version));
    json.attributeBegin("canonical_dataflow_ref");
    writeArtifactRoot(json, binary.canonicalDataflow());
    json.attributeEnd();
    json.attributeBegin("compiler_target_binding_ref");
    writeArtifactRoot(json, binary.compilerTargetBinding());
    json.attributeEnd();
    json.attribute("code_blob", formatBlobDigestHex(binary.codeBlob()));
    json.attributeArray("load_segments", [&] {
      for (const InstructionLoadSegment &segment : binary.loadSegments()) {
        json.object([&] {
          json.attribute("segment_ordinal", segment.ordinal);
          json.attribute("virtual_address", segment.virtualAddress);
          json.attribute("file_offset", segment.fileOffset);
          json.attribute("file_size", segment.fileSize);
          json.attribute("memory_size", segment.memorySize);
          json.attribute("alignment", segment.alignment);
          json.attribute("readable", segment.readable);
          json.attribute("writable", segment.writable);
          json.attribute("executable", segment.executable);
        });
      }
    });
    json.attributeArray("thread_entry_table", [&] {
      for (const ThreadEntryBinding &entry : binary.threadEntryTable()) {
        const std::vector<std::uint8_t> local =
            llvm::cantFail(dataflow::encodeDataflowReference(
                binary.canonicalDataflow().artifact, entry.rootThreadLaunch));
        json.object([&] {
          json.attribute("root_thread_launch_ref",
                         formatArtifactLocalPayloadHex(local));
          json.attribute("entry_ordinal", entry.entryOrdinal);
        });
      }
    });
    json.attributeArray("runtime_imports", [&] {
      for (const RuntimeImport &entry : binary.runtimeImports()) {
        json.object([&] {
          json.attribute("support_component_ordinal",
                         entry.supportComponentOrdinal);
          json.attribute("abi_symbol", entry.abiSymbol);
          if (entry.abiSymbolVersion)
            json.attribute("abi_symbol_version", *entry.abiSymbolVersion);
        });
      }
    });
  });
  return output.str().str();
}

llvm::Expected<DecodedInstructionCoreBinaryFields>
parseInstructionCoreBinaryFields(llvm::StringRef jsonText) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return codecError("instruction_core_binary_invalid",
                      "root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "InstructionCoreBinary",
          {"schema", "schema_version", "canonical_dataflow_ref",
           "compiler_target_binding_ref", "code_blob", "load_segments",
           "thread_entry_table", "runtime_imports"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "InstructionCoreBinary");
  if (!schema)
    return schema.takeError();
  if (*schema != instructionCoreBinarySchema.identity)
    return codecError("instruction_core_binary_schema_unsupported",
                      "unsupported schema '" + *schema + "'");
  auto version =
      requireString(*root, "schema_version", "InstructionCoreBinary");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (*parsedVersion != instructionCoreBinarySchema.version)
    return codecError("instruction_core_binary_schema_unsupported",
                      "unsupported schema_version '" + *version + "'");
  auto dataflow = parseArtifactRoot(*root, "canonical_dataflow_ref");
  if (!dataflow)
    return dataflow.takeError();
  auto target = parseArtifactRoot(*root, "compiler_target_binding_ref");
  if (!target)
    return target.takeError();
  auto blobText = requireString(*root, "code_blob", "InstructionCoreBinary");
  if (!blobText)
    return blobText.takeError();
  auto blob = parseBlobDigestHex(*blobText);
  if (!blob)
    return blob.takeError();
  auto segments = parseLoadSegments(*root);
  if (!segments)
    return segments.takeError();
  auto entries = parseThreadEntries(*root, dataflow->artifact);
  if (!entries)
    return entries.takeError();
  auto imports = parseRuntimeImports(*root);
  if (!imports)
    return imports.takeError();
  return DecodedInstructionCoreBinaryFields{
      std::move(*dataflow), std::move(*target),  *blob,
      std::move(*segments), std::move(*entries), std::move(*imports)};
}

} // namespace loom::detail

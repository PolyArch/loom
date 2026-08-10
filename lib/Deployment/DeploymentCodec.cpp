#include "DeploymentInternal.h"

#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <utility>

namespace loom::deployment::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deployment_invalid: " + message);
}

llvm::Error
rejectUnknownFields(const llvm::json::Object &object, llvm::StringRef context,
                    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (std::find(allowed.begin(), allowed.end(),
                  llvm::StringRef(entry.first)) == allowed.end())
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(entry.first) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto result = value->getAsString();
  if (!result)
    return invalid(context + " field '" + field + "' must be a string");
  return *result;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto result = value->getAsUINT64();
  if (!result)
    return invalid(context + " field '" + field +
                   "' must be an unsigned integer");
  return *result;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  const llvm::json::Object *result = value->getAsObject();
  if (!result)
    return invalid(context + " field '" + field + "' must be an object");
  return result;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  const llvm::json::Array *result = value->getAsArray();
  if (!result)
    return invalid(context + " field '" + field + "' must be an array");
  return result;
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.attribute("schema", reference.schemaIdentity);
  json.attribute("schema_version",
                 formatSchemaVersion(reference.schemaVersion));
  json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
}

llvm::Expected<ArtifactRootReference>
parseRootReference(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "schema_version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  auto version = requireString(object, "schema_version", context);
  auto artifact = requireString(object, "artifact", context);
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (!artifact)
    return artifact.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

llvm::Expected<ArtifactRootReference>
parseRootReferenceField(const llvm::json::Object &object, llvm::StringRef field,
                        llvm::StringRef context) {
  auto nested = requireObject(object, field, context);
  if (!nested)
    return nested.takeError();
  return parseRootReference(**nested, (context + "." + field).str());
}

void writeTypeArray(llvm::json::OStream &json, llvm::StringRef field,
                    llvm::ArrayRef<CanonicalTypeBytes> types) {
  json.attributeArray(field, [&] {
    for (const CanonicalTypeBytes &type : types)
      json.value(formatArtifactLocalPayloadHex(type));
  });
}

llvm::Expected<std::vector<CanonicalTypeBytes>>
parseTypeArray(const llvm::json::Object &object, llvm::StringRef field,
               llvm::StringRef context) {
  auto values = requireArray(object, field, context);
  if (!values)
    return values.takeError();
  std::vector<CanonicalTypeBytes> result;
  result.reserve((*values)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**values)) {
    auto spelling = value.getAsString();
    if (!spelling)
      return invalid(context + "." + field + "[" + llvm::Twine(ordinal) +
                     "] must be a string");
    auto bytes = parseArtifactLocalPayloadHex(*spelling);
    if (!bytes)
      return bytes.takeError();
    result.push_back(std::move(*bytes));
  }
  return result;
}

llvm::StringRef interfaceKindName(HostExternalInterfaceKind kind) {
  switch (kind) {
  case HostExternalInterfaceKind::Value:
    return "value";
  case HostExternalInterfaceKind::Stream:
    return "stream";
  case HostExternalInterfaceKind::Memory:
    return "memory";
  }
  llvm_unreachable("unknown Host interface kind");
}

llvm::Expected<HostExternalInterfaceKind>
parseInterfaceKind(llvm::StringRef spelling) {
  if (spelling == "value")
    return HostExternalInterfaceKind::Value;
  if (spelling == "stream")
    return HostExternalInterfaceKind::Stream;
  if (spelling == "memory")
    return HostExternalInterfaceKind::Memory;
  return invalid("host external interface has an unknown kind");
}

llvm::StringRef interfaceDirectionName(HostExternalInterfaceDirection kind) {
  switch (kind) {
  case HostExternalInterfaceDirection::Input:
    return "input";
  case HostExternalInterfaceDirection::Output:
    return "output";
  case HostExternalInterfaceDirection::InOut:
    return "inout";
  }
  llvm_unreachable("unknown Host interface direction");
}

llvm::Expected<HostExternalInterfaceDirection>
parseInterfaceDirection(llvm::StringRef spelling) {
  if (spelling == "input")
    return HostExternalInterfaceDirection::Input;
  if (spelling == "output")
    return HostExternalInterfaceDirection::Output;
  if (spelling == "inout")
    return HostExternalInterfaceDirection::InOut;
  return invalid("host external interface has an unknown direction");
}

void writeHostProgram(llvm::json::OStream &json,
                      const HostProgramLeaf &program) {
  json.attributeObject("compiler_target_binding_ref", [&] {
    writeRootReference(json, program.compilerTargetBinding());
  });
  json.attribute("program_blob", formatBlobDigestHex(program.programBlob()));
  json.attributeArray("program_entries", [&] {
    for (const HostProgramEntry &entry : program.programEntries())
      json.object([&] {
        json.attribute("entry_ordinal", entry.entryOrdinal);
        json.attribute("abi_symbol", entry.abiSymbol);
        writeTypeArray(json, "value_argument_types", entry.valueArgumentTypes);
        writeTypeArray(json, "value_result_types", entry.valueResultTypes);
        json.attributeArray("external_interface_ordinals", [&] {
          for (std::uint64_t ordinal : entry.externalInterfaceOrdinals)
            json.value(ordinal);
        });
      });
  });
  json.attributeArray("external_interfaces", [&] {
    for (const HostExternalInterface &interface : program.externalInterfaces())
      json.object([&] {
        json.attribute("interface_ordinal", interface.interfaceOrdinal);
        json.attribute("kind", interfaceKindName(interface.kind));
        json.attribute("direction",
                       interfaceDirectionName(interface.direction));
        json.attribute("semantic_type",
                       formatArtifactLocalPayloadHex(interface.semanticType));
      });
  });
  json.attribute("registration_table_digest",
                 formatBlobDigestHex(program.registrationTableDigest()));
  json.attributeArray("support_component_ordinals", [&] {
    for (std::uint64_t ordinal : program.supportComponentOrdinals())
      json.value(ordinal);
  });
}

llvm::Expected<HostProgramLeaf>
parseHostProgram(const llvm::json::Object &object) {
  constexpr llvm::StringLiteral context = "host_program";
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"compiler_target_binding_ref", "program_blob", "program_entries",
           "external_interfaces", "registration_table_digest",
           "support_component_ordinals"}))
    return std::move(error);
  auto target =
      parseRootReferenceField(object, "compiler_target_binding_ref", context);
  auto programDigest = requireString(object, "program_blob", context);
  auto entries = requireArray(object, "program_entries", context);
  auto interfaces = requireArray(object, "external_interfaces", context);
  auto registration =
      requireString(object, "registration_table_digest", context);
  auto support = requireArray(object, "support_component_ordinals", context);
  if (!target)
    return target.takeError();
  if (!programDigest)
    return programDigest.takeError();
  if (!entries)
    return entries.takeError();
  if (!interfaces)
    return interfaces.takeError();
  if (!registration)
    return registration.takeError();
  if (!support)
    return support.takeError();
  auto parsedProgramDigest = parseBlobDigestHex(*programDigest);
  auto parsedRegistration = parseBlobDigestHex(*registration);
  if (!parsedProgramDigest)
    return parsedProgramDigest.takeError();
  if (!parsedRegistration)
    return parsedRegistration.takeError();

  std::vector<HostProgramEntry> parsedEntries;
  parsedEntries.reserve((*entries)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**entries)) {
    const llvm::json::Object *entry = value.getAsObject();
    const std::string itemContext =
        (context + ".program_entries[" + llvm::Twine(ordinal) + "]").str();
    if (!entry)
      return invalid(itemContext + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *entry, itemContext,
            {"entry_ordinal", "abi_symbol", "value_argument_types",
             "value_result_types", "external_interface_ordinals"}))
      return std::move(error);
    auto entryOrdinal = requireUnsigned(*entry, "entry_ordinal", itemContext);
    auto symbol = requireString(*entry, "abi_symbol", itemContext);
    auto arguments =
        parseTypeArray(*entry, "value_argument_types", itemContext);
    auto results = parseTypeArray(*entry, "value_result_types", itemContext);
    auto interfaceOrdinals =
        requireArray(*entry, "external_interface_ordinals", itemContext);
    if (!entryOrdinal)
      return entryOrdinal.takeError();
    if (!symbol)
      return symbol.takeError();
    if (!arguments)
      return arguments.takeError();
    if (!results)
      return results.takeError();
    if (!interfaceOrdinals)
      return interfaceOrdinals.takeError();
    std::vector<std::uint64_t> parsedOrdinals;
    parsedOrdinals.reserve((*interfaceOrdinals)->size());
    for (const llvm::json::Value &raw : **interfaceOrdinals) {
      auto parsed = raw.getAsUINT64();
      if (!parsed)
        return invalid(itemContext +
                       ".external_interface_ordinals must contain unsigned "
                       "integers");
      parsedOrdinals.push_back(*parsed);
    }
    parsedEntries.push_back({*entryOrdinal, symbol->str(),
                             std::move(*arguments), std::move(*results),
                             std::move(parsedOrdinals)});
  }

  std::vector<HostExternalInterface> parsedInterfaces;
  parsedInterfaces.reserve((*interfaces)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**interfaces)) {
    const llvm::json::Object *interface = value.getAsObject();
    const std::string itemContext =
        (context + ".external_interfaces[" + llvm::Twine(ordinal) + "]").str();
    if (!interface)
      return invalid(itemContext + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *interface, itemContext,
            {"interface_ordinal", "kind", "direction", "semantic_type"}))
      return std::move(error);
    auto interfaceOrdinal =
        requireUnsigned(*interface, "interface_ordinal", itemContext);
    auto kind = requireString(*interface, "kind", itemContext);
    auto direction = requireString(*interface, "direction", itemContext);
    auto type = requireString(*interface, "semantic_type", itemContext);
    if (!interfaceOrdinal)
      return interfaceOrdinal.takeError();
    if (!kind)
      return kind.takeError();
    if (!direction)
      return direction.takeError();
    if (!type)
      return type.takeError();
    auto parsedKind = parseInterfaceKind(*kind);
    auto parsedDirection = parseInterfaceDirection(*direction);
    auto parsedType = parseArtifactLocalPayloadHex(*type);
    if (!parsedKind)
      return parsedKind.takeError();
    if (!parsedDirection)
      return parsedDirection.takeError();
    if (!parsedType)
      return parsedType.takeError();
    parsedInterfaces.push_back({*interfaceOrdinal, *parsedKind,
                                *parsedDirection, std::move(*parsedType)});
  }

  std::vector<std::uint64_t> supportOrdinals;
  supportOrdinals.reserve((*support)->size());
  for (const llvm::json::Value &raw : **support) {
    auto ordinal = raw.getAsUINT64();
    if (!ordinal)
      return invalid("support_component_ordinals must contain unsigned "
                     "integers");
    supportOrdinals.push_back(*ordinal);
  }
  return DeploymentCodecAccess::hostProgram(
      std::move(*target), *parsedProgramDigest, std::move(parsedEntries),
      std::move(parsedInterfaces), *parsedRegistration,
      std::move(supportOrdinals));
}

llvm::StringRef permissionsName(frontend::StaticMemoryPermissions value) {
  switch (value) {
  case frontend::StaticMemoryPermissions::ReadOnly:
    return "read_only";
  case frontend::StaticMemoryPermissions::ReadWrite:
    return "read_write";
  }
  llvm_unreachable("unknown static memory permissions");
}

llvm::Expected<frontend::StaticMemoryPermissions>
parsePermissions(llvm::StringRef spelling) {
  if (spelling == "read_only")
    return frontend::StaticMemoryPermissions::ReadOnly;
  if (spelling == "read_write")
    return frontend::StaticMemoryPermissions::ReadWrite;
  return invalid("static memory image has unknown permissions");
}

void writeStaticMemory(llvm::json::OStream &json,
                       const StaticMemoryImageLeaf &memory) {
  json.attributeObject("canonical_dataflow_ref", [&] {
    writeRootReference(json, memory.canonicalDataflow());
  });
  auto root = dataflow::encodeDataflowReference(
      memory.canonicalDataflow().artifact, memory.logicalMemoryRoot());
  if (!root)
    llvm_unreachable("validated static memory root cannot fail encoding");
  json.attribute("logical_memory_root_ref",
                 formatArtifactLocalPayloadHex(*root));
  json.attributeObject("layout_binding_ref", [&] {
    writeRootReference(json, memory.layoutBinding());
  });
  json.attribute("size_bytes", memory.sizeBytes());
  json.attribute("alignment_bytes", memory.alignmentBytes());
  json.attribute("permissions", permissionsName(memory.permissions()));
  json.attributeArray("initialized_chunks", [&] {
    for (const StaticMemoryInitializedChunk &chunk : memory.initializedChunks())
      json.object([&] {
        json.attribute("byte_offset", chunk.byteOffset);
        json.attribute("byte_count", chunk.byteCount);
        json.attribute("blob_digest", formatBlobDigestHex(chunk.blobDigest));
      });
  });
  json.attributeArray("zero_fill_ranges", [&] {
    for (const StaticMemoryZeroFillRange &range : memory.zeroFillRanges())
      json.object([&] {
        json.attribute("byte_offset", range.byteOffset);
        json.attribute("byte_count", range.byteCount);
      });
  });
}

llvm::Expected<StaticMemoryImageLeaf>
parseStaticMemory(const llvm::json::Object &object, std::size_t ordinal) {
  const std::string context =
      ("static_memory_images[" + llvm::Twine(ordinal) + "]").str();
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"canonical_dataflow_ref", "logical_memory_root_ref",
           "layout_binding_ref", "size_bytes", "alignment_bytes", "permissions",
           "initialized_chunks", "zero_fill_ranges"}))
    return std::move(error);
  auto dataflow =
      parseRootReferenceField(object, "canonical_dataflow_ref", context);
  auto root = requireString(object, "logical_memory_root_ref", context);
  auto layout = parseRootReferenceField(object, "layout_binding_ref", context);
  auto size = requireUnsigned(object, "size_bytes", context);
  auto alignment = requireUnsigned(object, "alignment_bytes", context);
  auto permissions = requireString(object, "permissions", context);
  auto chunks = requireArray(object, "initialized_chunks", context);
  auto zeroFill = requireArray(object, "zero_fill_ranges", context);
  if (!dataflow)
    return dataflow.takeError();
  if (!root)
    return root.takeError();
  if (!layout)
    return layout.takeError();
  if (!size)
    return size.takeError();
  if (!alignment)
    return alignment.takeError();
  if (!permissions)
    return permissions.takeError();
  if (!chunks)
    return chunks.takeError();
  if (!zeroFill)
    return zeroFill.takeError();
  auto rootBytes = parseArtifactLocalPayloadHex(*root);
  if (!rootBytes)
    return rootBytes.takeError();
  auto logicalRoot =
      dataflow::decodeDataflowReference<dataflow::LogicalMemoryRootRef>(
          *rootBytes, dataflow->artifact);
  if (!logicalRoot)
    return logicalRoot.takeError();
  auto parsedPermissions = parsePermissions(*permissions);
  if (!parsedPermissions)
    return parsedPermissions.takeError();

  std::vector<StaticMemoryInitializedChunk> parsedChunks;
  parsedChunks.reserve((*chunks)->size());
  for (const auto &[chunkOrdinal, value] : llvm::enumerate(**chunks)) {
    const llvm::json::Object *chunk = value.getAsObject();
    const std::string itemContext =
        (context + ".initialized_chunks[" + llvm::Twine(chunkOrdinal) + "]")
            .str();
    if (!chunk)
      return invalid(itemContext + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *chunk, itemContext, {"byte_offset", "byte_count", "blob_digest"}))
      return std::move(error);
    auto offset = requireUnsigned(*chunk, "byte_offset", itemContext);
    auto count = requireUnsigned(*chunk, "byte_count", itemContext);
    auto digest = requireString(*chunk, "blob_digest", itemContext);
    if (!offset)
      return offset.takeError();
    if (!count)
      return count.takeError();
    if (!digest)
      return digest.takeError();
    auto parsedDigest = parseBlobDigestHex(*digest);
    if (!parsedDigest)
      return parsedDigest.takeError();
    parsedChunks.push_back({*offset, *count, *parsedDigest});
  }

  std::vector<StaticMemoryZeroFillRange> parsedZeroFill;
  parsedZeroFill.reserve((*zeroFill)->size());
  for (const auto &[rangeOrdinal, value] : llvm::enumerate(**zeroFill)) {
    const llvm::json::Object *range = value.getAsObject();
    const std::string itemContext =
        (context + ".zero_fill_ranges[" + llvm::Twine(rangeOrdinal) + "]")
            .str();
    if (!range)
      return invalid(itemContext + " must be an object");
    if (llvm::Error error = rejectUnknownFields(*range, itemContext,
                                                {"byte_offset", "byte_count"}))
      return std::move(error);
    auto offset = requireUnsigned(*range, "byte_offset", itemContext);
    auto count = requireUnsigned(*range, "byte_count", itemContext);
    if (!offset)
      return offset.takeError();
    if (!count)
      return count.takeError();
    parsedZeroFill.push_back({*offset, *count});
  }
  return DeploymentCodecAccess::staticMemory(
      std::move(*dataflow), *logicalRoot, std::move(*layout), *size, *alignment,
      *parsedPermissions, std::move(parsedChunks), std::move(parsedZeroFill));
}

llvm::Expected<llvm::json::Value>
parseInlineImage(const llvm::json::Value &value,
                 const ArtifactSchemaDescriptor &schema,
                 llvm::StringRef context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  auto identity = requireString(*object, "schema", context);
  auto version = requireString(*object, "schema_version", context);
  if (!identity)
    return identity.takeError();
  if (!version)
    return version.takeError();
  if (*identity != schema.identity ||
      *version != formatSchemaVersion(schema.version))
    return invalid(context + " has the wrong schema descriptor");
  return value;
}

llvm::Expected<llvm::json::Value>
parseCanonicalInlineBytes(const CanonicalSemanticBytes &bytes,
                          const ArtifactSchemaDescriptor &schema) {
  llvm::StringRef text(reinterpret_cast<const char *>(bytes.bytes().data()),
                       bytes.bytes().size());
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return invalid(schema.identity + " is not valid JSON");
  return parseInlineImage(*parsed, schema, schema.identity);
}

} // namespace

HostProgramLeaf DeploymentCodecAccess::hostProgram(
    ArtifactRootReference compilerTargetBinding, BlobDigest programBlob,
    std::vector<HostProgramEntry> programEntries,
    std::vector<HostExternalInterface> externalInterfaces,
    BlobDigest registrationTableDigest,
    std::vector<std::uint64_t> supportComponentOrdinals) {
  return HostProgramLeaf(std::move(compilerTargetBinding), programBlob,
                         std::move(programEntries),
                         std::move(externalInterfaces), registrationTableDigest,
                         std::move(supportComponentOrdinals));
}

StaticMemoryImageLeaf DeploymentCodecAccess::staticMemory(
    ArtifactRootReference canonicalDataflow,
    dataflow::LogicalMemoryRootRef logicalMemoryRoot,
    ArtifactRootReference layoutBinding, std::uint64_t sizeBytes,
    std::uint64_t alignmentBytes, frontend::StaticMemoryPermissions permissions,
    std::vector<StaticMemoryInitializedChunk> initializedChunks,
    std::vector<StaticMemoryZeroFillRange> zeroFillRanges) {
  return StaticMemoryImageLeaf(
      std::move(canonicalDataflow), logicalMemoryRoot, std::move(layoutBinding),
      sizeBytes, alignmentBytes, permissions, std::move(initializedChunks),
      std::move(zeroFillRanges));
}

InlineRuntimeImage
DeploymentCodecAccess::runtimeImage(ArtifactSchemaDescriptor schema,
                                    CanonicalSemanticBytes bytes) {
  return InlineRuntimeImage(schema, std::move(bytes));
}

Deployment DeploymentCodecAccess::deployment(
    ArtifactRootReference systemMapping, HostProgramLeaf hostProgram,
    std::vector<ArtifactRootReference> instructionCoreBinaries,
    std::vector<DeploymentHardwareBinding> hardwareBindings,
    std::vector<ArtifactRootReference> configurationImages,
    std::vector<StaticMemoryImageLeaf> staticMemoryImages,
    InlineRuntimeImage threadDispatchImage,
    std::optional<InlineRuntimeImage> spatialLaunchImage,
    InlineRuntimeImage admissionImage) {
  return Deployment(std::move(systemMapping), std::move(hostProgram),
                    std::move(instructionCoreBinaries),
                    std::move(hardwareBindings), std::move(configurationImages),
                    std::move(staticMemoryImages),
                    std::move(threadDispatchImage),
                    std::move(spatialLaunchImage), std::move(admissionImage));
}

FinalizedDeployment
DeploymentCodecAccess::finalized(ArtifactRootReference reference,
                                 CanonicalSemanticBytes canonicalBytes,
                                 Deployment deployment) {
  return FinalizedDeployment(std::move(reference), std::move(canonicalBytes),
                             std::move(deployment));
}

llvm::Expected<ParsedDeployment>
parseDeployment(llvm::ArrayRef<std::uint8_t> bytes) {
  llvm::StringRef text(reinterpret_cast<const char *>(bytes.data()),
                       bytes.size());
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return invalid("canonical bytes are not valid JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("canonical bytes must contain a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "Deployment",
          {"schema", "schema_version", "system_mapping_ref", "host_program",
           "instruction_core_binary_refs", "hardware_bindings",
           "configuration_image_refs", "static_memory_images",
           "thread_dispatch_image", "spatial_launch_image", "admission_image"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "Deployment");
  auto version = requireString(*root, "schema_version", "Deployment");
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (*schema != deploymentSchema.identity ||
      *version != formatSchemaVersion(deploymentSchema.version))
    return invalid("root has the wrong schema descriptor");

  auto system =
      parseRootReferenceField(*root, "system_mapping_ref", "Deployment");
  auto hostObject = requireObject(*root, "host_program", "Deployment");
  auto binaries =
      requireArray(*root, "instruction_core_binary_refs", "Deployment");
  auto hardware = requireArray(*root, "hardware_bindings", "Deployment");
  auto images = requireArray(*root, "configuration_image_refs", "Deployment");
  auto staticMemory = requireArray(*root, "static_memory_images", "Deployment");
  const llvm::json::Value *thread = root->get("thread_dispatch_image");
  const llvm::json::Value *spatial = root->get("spatial_launch_image");
  const llvm::json::Value *admission = root->get("admission_image");
  if (!system)
    return system.takeError();
  if (!hostObject)
    return hostObject.takeError();
  if (!binaries)
    return binaries.takeError();
  if (!hardware)
    return hardware.takeError();
  if (!images)
    return images.takeError();
  if (!staticMemory)
    return staticMemory.takeError();
  if (!thread)
    return invalid("Deployment is missing field 'thread_dispatch_image'");
  if (!admission)
    return invalid("Deployment is missing field 'admission_image'");
  auto host = parseHostProgram(**hostObject);
  auto parsedThread = parseInlineImage(*thread, threadDispatchImageSchema,
                                       "thread_dispatch_image");
  auto parsedAdmission =
      parseInlineImage(*admission, admissionImageSchema, "admission_image");
  if (!host)
    return host.takeError();
  if (!parsedThread)
    return parsedThread.takeError();
  if (!parsedAdmission)
    return parsedAdmission.takeError();

  std::vector<ArtifactRootReference> parsedBinaries;
  parsedBinaries.reserve((*binaries)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**binaries)) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("instruction_core_binary_refs must contain objects");
    auto reference = parseRootReference(
        *object,
        ("instruction_core_binary_refs[" + llvm::Twine(ordinal) + "]").str());
    if (!reference)
      return reference.takeError();
    parsedBinaries.push_back(std::move(*reference));
  }

  std::vector<DeploymentHardwareBinding> parsedHardware;
  parsedHardware.reserve((*hardware)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**hardware)) {
    const llvm::json::Object *object = value.getAsObject();
    const std::string context =
        ("hardware_bindings[" + llvm::Twine(ordinal) + "]").str();
    if (!object)
      return invalid(context + " must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, context,
            {"hardware_implementation_ref", "runtime_platform_binding_ref"}))
      return std::move(error);
    auto implementation = parseRootReferenceField(
        *object, "hardware_implementation_ref", context);
    auto platform = parseRootReferenceField(
        *object, "runtime_platform_binding_ref", context);
    if (!implementation)
      return implementation.takeError();
    if (!platform)
      return platform.takeError();
    parsedHardware.push_back(
        {std::move(*implementation), std::move(*platform)});
  }

  std::vector<ArtifactRootReference> parsedImages;
  parsedImages.reserve((*images)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**images)) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("configuration_image_refs must contain objects");
    auto reference = parseRootReference(
        *object,
        ("configuration_image_refs[" + llvm::Twine(ordinal) + "]").str());
    if (!reference)
      return reference.takeError();
    parsedImages.push_back(std::move(*reference));
  }

  std::vector<StaticMemoryImageLeaf> parsedStaticMemory;
  parsedStaticMemory.reserve((*staticMemory)->size());
  for (const auto &[ordinal, value] : llvm::enumerate(**staticMemory)) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("static_memory_images must contain objects");
    auto memory = parseStaticMemory(*object, ordinal);
    if (!memory)
      return memory.takeError();
    parsedStaticMemory.push_back(std::move(*memory));
  }

  std::optional<llvm::json::Value> parsedSpatial;
  if (spatial) {
    auto value = parseInlineImage(*spatial, spatialLaunchImageSchema,
                                  "spatial_launch_image");
    if (!value)
      return value.takeError();
    parsedSpatial = std::move(*value);
  }
  return ParsedDeployment{
      std::move(*system),         std::move(*host),
      std::move(parsedBinaries),  std::move(parsedHardware),
      std::move(parsedImages),    std::move(parsedStaticMemory),
      std::move(*parsedThread),   std::move(parsedSpatial),
      std::move(*parsedAdmission)};
}

llvm::Expected<CanonicalSemanticBytes>
serializeDeployment(const ParsedDeployment &deployment,
                    const DerivedRuntimeImages &images) {
  auto thread = parseCanonicalInlineBytes(images.threadDispatch,
                                          threadDispatchImageSchema);
  auto admission =
      parseCanonicalInlineBytes(images.admission, admissionImageSchema);
  if (!thread)
    return thread.takeError();
  if (!admission)
    return admission.takeError();
  if (images.spatialLaunch) {
    auto spatial = parseCanonicalInlineBytes(*images.spatialLaunch,
                                             spatialLaunchImageSchema);
    if (!spatial)
      return spatial.takeError();
  }

  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", deploymentSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(deploymentSchema.version));
    json.attributeObject("system_mapping_ref", [&] {
      writeRootReference(json, deployment.systemMapping);
    });
    json.attributeObject("host_program", [&] {
      writeHostProgram(json, deployment.hostProgram);
    });
    json.attributeArray("instruction_core_binary_refs", [&] {
      for (const ArtifactRootReference &reference :
           deployment.instructionCoreBinaries)
        json.object([&] { writeRootReference(json, reference); });
    });
    json.attributeArray("hardware_bindings", [&] {
      for (const DeploymentHardwareBinding &binding :
           deployment.hardwareBindings)
        json.object([&] {
          json.attributeObject("hardware_implementation_ref", [&] {
            writeRootReference(json, binding.hardwareImplementation);
          });
          json.attributeObject("runtime_platform_binding_ref", [&] {
            writeRootReference(json, binding.runtimePlatformBinding);
          });
        });
    });
    json.attributeArray("configuration_image_refs", [&] {
      for (const ArtifactRootReference &reference :
           deployment.configurationImages)
        json.object([&] { writeRootReference(json, reference); });
    });
    json.attributeArray("static_memory_images", [&] {
      for (const StaticMemoryImageLeaf &memory : deployment.staticMemoryImages)
        json.object([&] { writeStaticMemory(json, memory); });
    });
    json.attributeBegin("thread_dispatch_image");
    json.rawValue(llvm::StringRef(
        reinterpret_cast<const char *>(images.threadDispatch.bytes().data()),
        images.threadDispatch.bytes().size()));
    json.attributeEnd();
    if (images.spatialLaunch) {
      json.attributeBegin("spatial_launch_image");
      json.rawValue(llvm::StringRef(
          reinterpret_cast<const char *>(images.spatialLaunch->bytes().data()),
          images.spatialLaunch->bytes().size()));
      json.attributeEnd();
    }
    json.attributeBegin("admission_image");
    json.rawValue(llvm::StringRef(
        reinterpret_cast<const char *>(images.admission.bytes().data()),
        images.admission.bytes().size()));
    json.attributeEnd();
  });
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(storage.begin(), storage.end()));
}

} // namespace loom::deployment::detail

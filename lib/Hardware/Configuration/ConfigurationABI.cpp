#include "Hardware/Configuration/ConfigurationABI.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware {
namespace {

using ByteVector = std::vector<std::uint8_t>;

thread_local detail::ConfigurationABIImportSessionState *
    currentConfigurationABIImportSession = nullptr;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "configuration_abi_invalid: " + message);
}

llvm::Error contextual(llvm::Error error, const llvm::Twine &context) {
  if (!error)
    return llvm::Error::success();
  return llvm::joinErrors(invalid(context), std::move(error));
}

template <typename T>
llvm::Expected<T> contextual(llvm::Expected<T> value,
                             const llvm::Twine &context) {
  if (!value)
    return contextual(value.takeError(), context);
  return std::move(*value);
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(entry.first)))
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(entry.first) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return invalid(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return invalid(context + " requires unsigned field '" + key + "'");
  std::optional<std::uint64_t> result = value->getAsUINT64();
  if (!result)
    return invalid(context + " field '" + key +
                   "' must be an unsigned integer");
  return *result;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return invalid(context + " requires object field '" + key + "'");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return invalid(context + " requires array field '" + key + "'");
  return value;
}

llvm::Expected<std::size_t> byteCountForBits(std::uint64_t bitCount,
                                             const llvm::Twine &context) {
  const std::uint64_t byteCount =
      bitCount / 8 + static_cast<std::uint64_t>((bitCount % 8) != 0);
  if (byteCount > std::numeric_limits<std::size_t>::max())
    return invalid(context + " is too large for this host");
  return static_cast<std::size_t>(byteCount);
}

llvm::Error validateBitVector(llvm::ArrayRef<std::uint8_t> bytes,
                              std::uint64_t bitCount,
                              const llvm::Twine &context) {
  auto expected = byteCountForBits(bitCount, context);
  if (!expected)
    return expected.takeError();
  if (bytes.size() != *expected)
    return invalid(context + " has the wrong byte count");
  const unsigned usedBits = static_cast<unsigned>(bitCount % 8);
  if (usedBits != 0 && !bytes.empty()) {
    const std::uint8_t unusedMask =
        static_cast<std::uint8_t>(0xffU << usedBits);
    if ((bytes.back() & unusedMask) != 0)
      return invalid(context + " has nonzero padding bits");
  }
  return llvm::Error::success();
}

bool bit(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t index) {
  return ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) !=
         0;
}

void setBit(std::vector<std::uint8_t> &bytes, std::uint64_t index, bool value) {
  const std::size_t byteIndex = static_cast<std::size_t>(index / 8);
  const std::uint8_t mask = static_cast<std::uint8_t>(1U << (index % 8));
  if (value)
    bytes[byteIndex] |= mask;
  else
    bytes[byteIndex] &= static_cast<std::uint8_t>(~mask);
}

llvm::Error validateRange(std::uint64_t offset, std::uint64_t count,
                          std::uint64_t limit, const llvm::Twine &description) {
  if (count == 0)
    return invalid(description + " has zero bit_count");
  if (offset > limit || count > limit - offset)
    return invalid(description + " is out of range");
  return llvm::Error::success();
}

template <typename Ref> ByteVector referenceKey(const Ref &reference) {
  return fabric::canonicalFabricBytes(reference);
}

llvm::Expected<fabric::FabricModulePhysicalOwnerRef>
modulePhysicalOwner(const fabric::FabricInventoryOwnerRef &owner) {
  using fabric::FabricInventoryOwnerKind;
  switch (owner.kind()) {
  case FabricInventoryOwnerKind::PeOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricPeOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::FuOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricFuOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::FuOccurrenceNode:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricFuOccurrenceNodeRef>(owner.payload));
  case FabricInventoryOwnerKind::MemoryOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricMemoryOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::MemoryOperationPort:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricMemoryOperationPortRef>(owner.payload));
  case FabricInventoryOwnerKind::MemoryService: {
    const auto &service =
        std::get<fabric::FabricMemoryServiceRef>(owner.payload);
    if (service.kind() != fabric::FabricMemoryServiceKind::Local)
      break;
    return fabric::FabricModulePhysicalOwnerRef::create(
        fabric::LocalMemoryServiceRef(service));
  }
  case FabricInventoryOwnerKind::SwitchOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricSwitchOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::FifoOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricFifoOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::BoundaryOccurrence:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::FabricBoundaryOccurrenceRef>(owner.payload));
  case FabricInventoryOwnerKind::InstructionContext:
    return fabric::FabricModulePhysicalOwnerRef::create(
        std::get<fabric::InstructionContextRef>(owner.payload));
  default:
    break;
  }
  return invalid("semantic field is not owned by a Module physical resource");
}

llvm::Expected<fabric::FabricPhysicalOccurrenceOwnerRef> physicalOwnerForField(
    const fabric::FabricPhysicalConfigurationFieldRef &field) {
  if (field.kind() ==
      fabric::FabricPhysicalConfigurationFieldKind::DirectSystemField) {
    const auto &local =
        std::get<fabric::FabricSemanticConfigFieldRef>(field.payload());
    return fabric::FabricPhysicalOccurrenceOwnerRef::create(
        local.owner.catalog());
  }

  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(field.payload());
  const auto &local =
      std::get<fabric::FabricSemanticConfigFieldRef>(internal.target.payload());
  auto localOwner = modulePhysicalOwner(local.owner.catalog());
  if (!localOwner)
    return localOwner.takeError();
  auto target = fabric::FabricModulePhysicalTargetRef::create(*localOwner);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalOccurrenceOwnerRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{internal.spatialCore,
                                               std::move(*target)});
}

fabric::FabricInventoryOwnerRef
inventoryOwner(const fabric::FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> fabric::FabricInventoryOwnerRef {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, fabric::LocalMemoryServiceRef>)
          return fabric::FabricInventoryOwnerRef::of(value.underlying());
        else
          return fabric::FabricInventoryOwnerRef::of(value);
      },
      owner.payload());
}

llvm::Expected<fabric::FabricArtifactView>
importedModuleFor(const fabric::FabricSystemRootView &system,
                  const fabric::SpatialCoreOccurrenceRef &spatialCore) {
  auto target = system.spatialCoreTarget(spatialCore.core);
  if (!target)
    return invalid("SpatialCore has no imported Module target");
  llvm::ArrayRef<fabric::FabricArtifactView> modules =
      system.artifact().importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalid("SpatialCore imported Module dependency is out of range");
  const fabric::FabricArtifactView &module =
      modules[static_cast<std::size_t>(target->dependencyOrdinal)];
  if (module.moduleRootTemplate() != target->target)
    return invalid("SpatialCore imported Module target does not match");
  return module;
}

llvm::Error validatePhysicalField(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricPhysicalConfigurationFieldRef &field) {
  if (field.kind() ==
      fabric::FabricPhysicalConfigurationFieldKind::DirectSystemField)
    return fabric::validateFabricRef(
        system.artifact(),
        std::get<fabric::FabricSemanticConfigFieldRef>(field.payload()));

  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(field.payload());
  auto module = importedModuleFor(system, internal.spatialCore);
  if (!module)
    return module.takeError();
  return fabric::validateFabricRef(
      *module, std::get<fabric::FabricSemanticConfigFieldRef>(
                   internal.target.payload()));
}

struct ResolvedPhysicalField final {
  fabric::FabricArtifactView artifact;
  fabric::FabricSemanticConfigFieldRef local;
};

inline constexpr std::uint64_t semanticValidationAlgorithmVersion = 1;

struct SemanticValidationKey final {
  ArtifactIdentity::Storage fabricArtifact;
  SchemaVersion abiSchemaVersion;
  std::uint64_t algorithmVersion = 0;
  ByteVector localField;
  ConfigurationEncodingRelationId encodingRelation = 0;

  friend bool operator<(const SemanticValidationKey &lhs,
                        const SemanticValidationKey &rhs) {
    return std::tie(lhs.fabricArtifact, lhs.abiSchemaVersion.major,
                    lhs.abiSchemaVersion.minor, lhs.algorithmVersion,
                    lhs.localField, lhs.encodingRelation) <
           std::tie(rhs.fabricArtifact, rhs.abiSchemaVersion.major,
                    rhs.abiSchemaVersion.minor, rhs.algorithmVersion,
                    rhs.localField, rhs.encodingRelation);
  }
};

llvm::Expected<ResolvedPhysicalField>
resolvePhysicalField(const fabric::FabricSystemRootView &system,
                     const fabric::FabricPhysicalConfigurationFieldRef &field) {
  if (field.kind() ==
      fabric::FabricPhysicalConfigurationFieldKind::DirectSystemField)
    return ResolvedPhysicalField{
        system.artifact(),
        std::get<fabric::FabricSemanticConfigFieldRef>(field.payload())};

  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(field.payload());
  auto module = importedModuleFor(system, internal.spatialCore);
  if (!module)
    return module.takeError();
  return ResolvedPhysicalField{std::move(*module),
                               std::get<fabric::FabricSemanticConfigFieldRef>(
                                   internal.target.payload())};
}

llvm::Error
validatePhysicalSlot(const fabric::FabricSystemRootView &system,
                     const fabric::FabricPhysicalConfigurationSlotRef &slot) {
  const auto field = fabric::configurationField(slot);
  if (llvm::Error error = validatePhysicalField(system, field))
    return error;
  auto resolved = resolvePhysicalField(system, field);
  if (!resolved)
    return resolved.takeError();
  return resolved->artifact.validateConfigurationSlot(
      fabric::configurationSlot(slot));
}

llvm::Expected<std::set<ByteVector>>
finiteSemanticDomain(const ResolvedPhysicalField &field,
                     mlir::MLIRContext &context) {
  auto relation = field.artifact.semanticFieldRelation(field.local, context);
  if (!relation)
    return relation.takeError();
  if (relation->kind() != fabric::FabricSemanticFieldRelationKind::Finite)
    return invalid("configuration field does not have a finite domain");
  std::set<ByteVector> result;
  for (const CanonicalSemanticBytes &value : relation->finiteDomain()) {
    if (!result.insert(ByteVector(value.bytes().begin(), value.bytes().end()))
             .second)
      return invalid("configuration domain has a duplicate value");
  }
  return result;
}

llvm::Error
validateSemanticValue(const fabric::FabricSystemRootView &system,
                      const fabric::FabricPhysicalConfigurationFieldRef &field,
                      llvm::ArrayRef<std::uint8_t> value) {
  auto resolved = resolvePhysicalField(system, field);
  if (!resolved)
    return resolved.takeError();
  mlir::MLIRContext context;
  auto relation =
      resolved->artifact.semanticFieldRelation(resolved->local, context);
  if (!relation)
    return relation.takeError();
  return relation->validateSemanticValue(value);
}

template <typename Ref>
llvm::Expected<Ref> parseReference(llvm::StringRef spelling,
                                   const llvm::Twine &context) {
  auto bytes = contextual(parseArtifactLocalPayloadHex(spelling), context);
  if (!bytes)
    return bytes.takeError();
  return contextual(fabric::decodeFabricRef<Ref>(*bytes), context);
}

llvm::Expected<ArtifactRootReference>
parseRootReference(const llvm::json::Object &object) {
  constexpr llvm::StringLiteral context = "fabric_ref";
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "schema_version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  if (!schema)
    return schema.takeError();
  auto version = requireString(object, "schema_version", context);
  if (!version)
    return version.takeError();
  auto parsedVersion = contextual(parseSchemaVersion(*version), context);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto artifact = requireString(object, "artifact", context);
  if (!artifact)
    return artifact.takeError();
  auto parsedArtifact =
      contextual(parseArtifactIdentityHex(*artifact), context);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

llvm::Expected<FiniteCodebookEntry>
parseCodebookEntry(const llvm::json::Value &value) {
  constexpr llvm::StringLiteral context = "finite codebook entry";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, context, {"semantic_value", "physical_code"}))
    return std::move(error);
  auto semantic = requireString(*object, "semantic_value", context);
  if (!semantic)
    return semantic.takeError();
  auto semanticBytes =
      contextual(parseArtifactLocalPayloadHex(*semantic), context);
  if (!semanticBytes)
    return semanticBytes.takeError();
  auto physical = requireString(*object, "physical_code", context);
  if (!physical)
    return physical.takeError();
  auto physicalBytes =
      contextual(parseArtifactLocalPayloadHex(*physical), context);
  if (!physicalBytes)
    return physicalBytes.takeError();
  return FiniteCodebookEntry{std::move(*semanticBytes),
                             std::move(*physicalBytes)};
}

llvm::Expected<SemanticFieldEncoding>
parseSemanticEncoding(const llvm::json::Object &object) {
  constexpr llvm::StringLiteral context = "semantic_encoding";
  auto kind = requireString(object, "kind", context);
  if (!kind)
    return kind.takeError();
  if (*kind == "direct_bits") {
    if (llvm::Error error =
            rejectUnknownFields(object, context, {"kind", "encoded_bit_count"}))
      return std::move(error);
    auto bitCount = requireUnsigned(object, "encoded_bit_count", context);
    if (!bitCount)
      return bitCount.takeError();
    return SemanticFieldEncoding{DirectBitsEncoding{*bitCount}};
  }
  if (*kind != "finite_codebook")
    return invalid(context + " has unknown kind '" + *kind + "'");
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"kind", "encoded_bit_count", "entries"}))
    return std::move(error);
  auto bitCount = requireUnsigned(object, "encoded_bit_count", context);
  if (!bitCount)
    return bitCount.takeError();
  auto entries = requireArray(object, "entries", context);
  if (!entries)
    return entries.takeError();
  std::vector<FiniteCodebookEntry> parsedEntries;
  parsedEntries.reserve((*entries)->size());
  for (const llvm::json::Value &entry : **entries) {
    auto parsed = parseCodebookEntry(entry);
    if (!parsed)
      return parsed.takeError();
    parsedEntries.push_back(std::move(*parsed));
  }
  return SemanticFieldEncoding{
      FiniteCodebookEncoding{*bitCount, std::move(parsedEntries)}};
}

llvm::Expected<DestinationSlice>
parseDestinationSlice(const llvm::json::Value &value) {
  constexpr llvm::StringLiteral context = "destination slice";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, context,
          {"source_bit_offset", "destination_bit_offset", "bit_count"}))
    return std::move(error);
  auto source = requireUnsigned(*object, "source_bit_offset", context);
  if (!source)
    return source.takeError();
  auto destination =
      requireUnsigned(*object, "destination_bit_offset", context);
  if (!destination)
    return destination.takeError();
  auto count = requireUnsigned(*object, "bit_count", context);
  if (!count)
    return count.takeError();
  return DestinationSlice{*source, *destination, *count};
}

llvm::Expected<ConfigurationEncodingRelationDraft>
parseEncodingRelation(const llvm::json::Value &value,
                      std::uint64_t expectedId) {
  constexpr llvm::StringLiteral context = "configuration encoding relation";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, context,
          {"relation_id", "semantic_encoding", "inactive_value"}))
    return std::move(error);
  auto id = requireUnsigned(*object, "relation_id", context);
  if (!id)
    return id.takeError();
  if (*id != expectedId)
    return invalid(context + " relation_id is not dense canonical order");
  auto encodingObject = requireObject(*object, "semantic_encoding", context);
  if (!encodingObject)
    return encodingObject.takeError();
  auto encoding = parseSemanticEncoding(**encodingObject);
  if (!encoding)
    return encoding.takeError();
  auto inactive = requireString(*object, "inactive_value", context);
  if (!inactive)
    return inactive.takeError();
  auto inactiveBytes =
      contextual(parseArtifactLocalPayloadHex(*inactive), context);
  if (!inactiveBytes)
    return inactiveBytes.takeError();
  return ConfigurationEncodingRelationDraft{std::move(*encoding),
                                            std::move(*inactiveBytes)};
}

llvm::Expected<ConfigurationFieldEncoding>
parseField(const llvm::json::Value &value) {
  constexpr llvm::StringLiteral context = "configuration field";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, context,
                              {"fabric_config_slot_ref", "encoding_relation_id",
                               "destination_slices"}))
    return std::move(error);
  auto slotText = requireString(*object, "fabric_config_slot_ref", context);
  if (!slotText)
    return slotText.takeError();
  auto slot = parseReference<fabric::FabricPhysicalConfigurationSlotRef>(
      *slotText, "fabric_config_slot_ref");
  if (!slot)
    return slot.takeError();
  auto relationId = requireUnsigned(*object, "encoding_relation_id", context);
  if (!relationId)
    return relationId.takeError();
  auto slices = requireArray(*object, "destination_slices", context);
  if (!slices)
    return slices.takeError();
  std::vector<DestinationSlice> parsedSlices;
  parsedSlices.reserve((*slices)->size());
  for (const llvm::json::Value &slice : **slices) {
    auto parsed = parseDestinationSlice(slice);
    if (!parsed)
      return parsed.takeError();
    parsedSlices.push_back(*parsed);
  }
  return ConfigurationFieldEncoding{std::move(*slot), *relationId,
                                    std::move(parsedSlices)};
}

llvm::Expected<ProgrammingUnitDraft>
parseProgrammingUnit(const llvm::json::Value &value, std::uint64_t expectedId) {
  constexpr llvm::StringLiteral context = "programming unit";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, context,
                              {"unit_id", "exact_fabric_resource_closure",
                               "programming_model", "fields"}))
    return std::move(error);
  auto id = requireUnsigned(*object, "unit_id", context);
  if (!id)
    return id.takeError();
  if (*id != expectedId)
    return invalid(context + " unit_id is not dense canonical order");
  auto closure =
      requireArray(*object, "exact_fabric_resource_closure", context);
  if (!closure)
    return closure.takeError();
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> parsedClosure;
  parsedClosure.reserve((*closure)->size());
  for (const llvm::json::Value &entry : **closure) {
    std::optional<llvm::StringRef> spelling = entry.getAsString();
    if (!spelling)
      return invalid("resource closure entry must be a string");
    auto reference = parseReference<fabric::FabricPhysicalOccurrenceOwnerRef>(
        *spelling, "resource closure entry");
    if (!reference)
      return reference.takeError();
    parsedClosure.push_back(std::move(*reference));
  }
  auto model = requireObject(*object, "programming_model", context);
  if (!model)
    return model.takeError();
  if (llvm::Error error = rejectUnknownFields(**model, "programming_model",
                                              {"kind", "payload_bit_count"}))
    return std::move(error);
  auto kind = requireString(**model, "kind", "programming_model");
  if (!kind)
    return kind.takeError();
  if (*kind != "complete_image_atomic")
    return invalid("programming_model has unknown kind '" + *kind + "'");
  auto payloadBitCount =
      requireUnsigned(**model, "payload_bit_count", "programming_model");
  if (!payloadBitCount)
    return payloadBitCount.takeError();
  auto fields = requireArray(*object, "fields", context);
  if (!fields)
    return fields.takeError();
  std::vector<ConfigurationFieldEncoding> parsedFields;
  parsedFields.reserve((*fields)->size());
  for (const llvm::json::Value &field : **fields) {
    auto parsed = parseField(field);
    if (!parsed)
      return parsed.takeError();
    parsedFields.push_back(std::move(*parsed));
  }
  return ProgrammingUnitDraft{std::move(parsedClosure), *payloadBitCount,
                              std::move(parsedFields)};
}

llvm::Expected<ConfigurationABIDraft>
parseConfigurationABI(llvm::ArrayRef<std::uint8_t> bytes) {
  llvm::StringRef text(reinterpret_cast<const char *>(bytes.data()),
                       bytes.size());
  auto value = llvm::json::parse(text);
  if (!value)
    return contextual(value.takeError(), "canonical JSON cannot be parsed");
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return invalid("root must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*root, "ConfigurationABI",
                              {"schema", "schema_version", "fabric_ref",
                               "encoding_relations", "programming_units"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "ConfigurationABI");
  if (!schema)
    return schema.takeError();
  if (*schema != configurationAbiSchema.identity)
    return invalid("unsupported schema '" + *schema + "'");
  auto version = requireString(*root, "schema_version", "ConfigurationABI");
  if (!version)
    return version.takeError();
  if (*version != formatSchemaVersion(configurationAbiSchema.version))
    return invalid("unsupported schema_version '" + *version + "'");
  auto fabricObject = requireObject(*root, "fabric_ref", "ConfigurationABI");
  if (!fabricObject)
    return fabricObject.takeError();
  auto fabricReference = parseRootReference(**fabricObject);
  if (!fabricReference)
    return fabricReference.takeError();
  auto relations =
      requireArray(*root, "encoding_relations", "ConfigurationABI");
  if (!relations)
    return relations.takeError();
  std::vector<ConfigurationEncodingRelationDraft> parsedRelations;
  parsedRelations.reserve((*relations)->size());
  std::uint64_t expectedRelationId = 0;
  for (const llvm::json::Value &relation : **relations) {
    auto parsed = parseEncodingRelation(relation, expectedRelationId++);
    if (!parsed)
      return parsed.takeError();
    parsedRelations.push_back(std::move(*parsed));
  }
  auto units = requireArray(*root, "programming_units", "ConfigurationABI");
  if (!units)
    return units.takeError();
  std::vector<ProgrammingUnitDraft> parsedUnits;
  parsedUnits.reserve((*units)->size());
  std::uint64_t expectedId = 0;
  for (const llvm::json::Value &unit : **units) {
    auto parsed = parseProgrammingUnit(unit, expectedId++);
    if (!parsed)
      return parsed.takeError();
    parsedUnits.push_back(std::move(*parsed));
  }
  return ConfigurationABIDraft{std::move(*fabricReference),
                               std::move(parsedRelations),
                               std::move(parsedUnits)};
}

llvm::Error canonicalizeEncoding(ConfigurationEncodingRelationDraft &relation) {
  const std::uint64_t bitCount = relation.encodedBitCount();
  if (bitCount == 0)
    return invalid("semantic encoding has zero encoded_bit_count");
  if (auto *direct =
          std::get_if<DirectBitsEncoding>(&relation.semanticEncoding)) {
    (void)direct;
    return validateBitVector(relation.inactiveValue, bitCount,
                             "DirectBits inactive value");
  }

  auto &codebook = std::get<FiniteCodebookEncoding>(relation.semanticEncoding);
  if (codebook.entries.empty())
    return invalid("finite codebook has no entries");
  for (const FiniteCodebookEntry &entry : codebook.entries) {
    if (llvm::Error error = validateBitVector(entry.physicalCode, bitCount,
                                              "finite codebook physical code"))
      return error;
  }
  llvm::sort(codebook.entries, [](const FiniteCodebookEntry &lhs,
                                  const FiniteCodebookEntry &rhs) {
    return std::tie(lhs.semanticValue, lhs.physicalCode) <
           std::tie(rhs.semanticValue, rhs.physicalCode);
  });
  std::set<ByteVector> physicalCodes;
  for (std::size_t index = 0; index < codebook.entries.size(); ++index) {
    if (index != 0 && codebook.entries[index - 1].semanticValue ==
                          codebook.entries[index].semanticValue)
      return invalid("finite codebook contains a duplicate semantic value");
    if (!physicalCodes.insert(codebook.entries[index].physicalCode).second)
      return invalid("finite codebook contains a duplicate physical code");
  }
  const auto inactive =
      llvm::find_if(codebook.entries, [&](const FiniteCodebookEntry &entry) {
        return entry.semanticValue == relation.inactiveValue;
      });
  if (inactive == codebook.entries.end())
    return invalid("finite codebook cannot encode its inactive value");
  return llvm::Error::success();
}

bool sameSemanticEncoding(const SemanticFieldEncoding &lhs,
                          const SemanticFieldEncoding &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<DirectBitsEncoding>(&lhs))
    return left->encodedBitCount ==
           std::get<DirectBitsEncoding>(rhs).encodedBitCount;
  const auto &left = std::get<FiniteCodebookEncoding>(lhs);
  const auto &right = std::get<FiniteCodebookEncoding>(rhs);
  return left.encodedBitCount == right.encodedBitCount &&
         left.entries == right.entries;
}

llvm::Error validateFieldEncoding(
    const ResolvedPhysicalField &resolved,
    const ConfigurationEncodingRelationDraft &encodingRelation,
    mlir::MLIRContext &context) {
  auto relation =
      resolved.artifact.semanticFieldRelation(resolved.local, context);
  if (!relation)
    return relation.takeError();
  switch (relation->kind()) {
  case fabric::FabricSemanticFieldRelationKind::None:
    return invalid("field is present for a fixed Fabric resource");
  case fabric::FabricSemanticFieldRelationKind::Finite: {
    const auto *codebook =
        std::get_if<FiniteCodebookEncoding>(&encodingRelation.semanticEncoding);
    if (!codebook)
      return invalid("finite Fabric field requires a finite codebook");
    auto expected = finiteSemanticDomain(resolved, context);
    if (!expected)
      return expected.takeError();
    std::set<ByteVector> actual;
    for (const FiniteCodebookEntry &entry : codebook->entries) {
      if (llvm::Error error =
              relation->validateSemanticValue(entry.semanticValue))
        return error;
      actual.insert(entry.semanticValue);
    }
    if (actual != *expected)
      return invalid("finite codebook does not equal its Fabric relation");

    const auto &owner = resolved.local.owner.catalog();
    if (owner.kind() == fabric::FabricInventoryOwnerKind::PeOccurrence) {
      auto schema = resolved.artifact.spatialPeConfigurationSchema(
          std::get<fabric::FabricPeOccurrenceRef>(owner.payload));
      if (!schema)
        return schema.takeError();
      const auto descriptor =
          llvm::find_if(schema->fields(), [&](const auto &candidate) {
            return candidate.reference == resolved.local;
          });
      if (descriptor == schema->fields().end())
        return invalid("PE configuration field is absent from its schema");
      if (descriptor->kind ==
          fabric::FabricPeConfigurationFieldKind::Activation) {
        auto disabled =
            schema->encode(resolved.local, fabric::FabricPeDisabled{});
        if (!disabled)
          return disabled.takeError();
        if (!disabled->bytes().equals(encodingRelation.inactiveValue))
          return invalid("PE activation inactive value is not Disabled");
      }
    }
    return llvm::Error::success();
  }
  case fabric::FabricSemanticFieldRelationKind::Direct: {
    const auto *direct =
        std::get_if<DirectBitsEncoding>(&encodingRelation.semanticEncoding);
    if (!direct)
      return invalid("direct Fabric field requires DirectBits");
    if (!relation->directEncodedBitCount() ||
        direct->encodedBitCount != *relation->directEncodedBitCount())
      return invalid("DirectBits width does not equal its Fabric relation");
    return relation->validateSemanticValue(encodingRelation.inactiveValue);
  }
  }
  llvm_unreachable("unknown Fabric semantic field relation kind");
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyModuleField(const fabric::SpatialCoreOccurrenceRef &spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(field);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

llvm::Expected<std::set<ByteVector>>
expectedPhysicalSlots(const fabric::FabricSystemRootView &system) {
  std::set<ByteVector> result;
  const auto append =
      [&](const fabric::FabricPhysicalConfigurationSlotRef &slot)
      -> llvm::Error {
    if (!result.insert(referenceKey(slot)).second)
      return invalid("Fabric physical configuration slot is duplicated");
    return llvm::Error::success();
  };

  const auto appendOwner = [&](const fabric::FabricArtifactView &view,
                               const fabric::FabricInventoryOwnerRef &owner,
                               const auto &qualify) -> llvm::Error {
    const std::uint64_t count = view.inventorySize(
        owner, fabric::FabricInventoryKind::SemanticConfigField);
    for (fabric::FabricOrdinal ordinal = 0; ordinal < count; ++ordinal) {
      fabric::FabricSemanticConfigFieldRef local{
          fabric::FabricConfigurationOwnerRef(owner), ordinal};
      auto residencies = view.configurationResidencies(local);
      if (!residencies)
        return residencies.takeError();
      for (const fabric::FabricConfigurationResidency &residency :
           *residencies) {
        auto physical =
            qualify(fabric::FabricConfigurationSlotRef{local, residency});
        if (!physical)
          return physical.takeError();
        if (llvm::Error error = append(*physical))
          return error;
      }
    }
    return llvm::Error::success();
  };

  std::vector<fabric::FabricInventoryOwnerRef> directOwners;
  const auto &artifact = system.artifact();
  for (auto owner : artifact.hostCoreOccurrences())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
  for (auto owner : artifact.accCoreOccurrences()) {
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(
        fabric::InstructionCoreContextRef{owner}));
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(
        fabric::SpatialCoreOccurrenceRef{owner}));
  }
  for (auto owner : artifact.systemMemoryServices())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(
        fabric::FabricMemoryServiceRef::system(owner)));
  for (auto owner : artifact.systemServiceEndpoints())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
  for (auto owner : artifact.systemServiceTransforms())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
  for (auto owner : system.transportResources()) {
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
    for (auto pattern : system.transferPatterns(owner))
      directOwners.push_back(fabric::FabricInventoryOwnerRef::of(pattern));
  }
  for (auto owner : system.hardwareDomains())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));
  for (auto owner : artifact.externalBoundaries())
    directOwners.push_back(fabric::FabricInventoryOwnerRef::of(owner));

  for (const fabric::FabricInventoryOwnerRef &owner : directOwners) {
    if (llvm::Error error = appendOwner(
            artifact, owner,
            [](const fabric::FabricConfigurationSlotRef &local) {
              return fabric::FabricPhysicalConfigurationSlotRef::create(local);
            }))
      return std::move(error);
  }

  for (fabric::AccCoreOccurrenceRef core : artifact.accCoreOccurrences()) {
    const fabric::SpatialCoreOccurrenceRef spatialCore{core};
    auto module = importedModuleFor(system, spatialCore);
    if (!module)
      return module.takeError();
    for (const fabric::FabricModuleDomainMemberRef &member :
         module->moduleDomainMembers()) {
      if (member.kind() != fabric::FabricModuleDomainMemberKind::Internal)
        continue;
      const auto &physicalOwner =
          std::get<fabric::FabricModulePhysicalOwnerRef>(member.payload);
      const fabric::FabricInventoryOwnerRef owner =
          inventoryOwner(physicalOwner);
      if (llvm::Error error = appendOwner(
              *module, owner,
              [&](const fabric::FabricConfigurationSlotRef &local) {
                return fabric::FabricPhysicalConfigurationSlotRef::create(
                    fabric::SpatialCoreInternalConfigurationSlotRef{spatialCore,
                                                                    local});
              }))
        return std::move(error);
    }
  }
  return result;
}

llvm::Error canonicalizeSlices(
    ConfigurationFieldEncoding &field, std::uint64_t sourceLimit,
    std::uint64_t payloadBitCount,
    std::vector<std::pair<std::uint64_t, std::uint64_t>> &destinationRanges) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> sourceRanges;
  sourceRanges.reserve(field.destinationSlices.size());
  for (const DestinationSlice &slice : field.destinationSlices) {
    if (llvm::Error error = validateRange(slice.sourceBitOffset, slice.bitCount,
                                          sourceLimit, "source bit slice"))
      return error;
    if (llvm::Error error =
            validateRange(slice.destinationBitOffset, slice.bitCount,
                          payloadBitCount, "destination bit slice"))
      return error;
    sourceRanges.emplace_back(slice.sourceBitOffset,
                              slice.sourceBitOffset + slice.bitCount);
    destinationRanges.emplace_back(slice.destinationBitOffset,
                                   slice.destinationBitOffset + slice.bitCount);
  }
  llvm::sort(sourceRanges);
  std::uint64_t cursor = 0;
  for (const auto &[begin, end] : sourceRanges) {
    if (begin != cursor)
      return invalid("source bits are not covered exactly once");
    cursor = end;
  }
  if (cursor != sourceLimit)
    return invalid("source bits are not covered exactly once");
  llvm::sort(field.destinationSlices, [](const DestinationSlice &lhs,
                                         const DestinationSlice &rhs) {
    return std::tie(lhs.destinationBitOffset, lhs.sourceBitOffset,
                    lhs.bitCount) < std::tie(rhs.destinationBitOffset,
                                             rhs.sourceBitOffset, rhs.bitCount);
  });
  return llvm::Error::success();
}

using ClosureKey = std::vector<ByteVector>;

ClosureKey closureKey(const ProgrammingUnit &unit) {
  ClosureKey result;
  result.reserve(unit.exactFabricResourceClosure.size());
  for (const fabric::FabricPhysicalOccurrenceOwnerRef &owner :
       unit.exactFabricResourceClosure)
    result.push_back(referenceKey(owner));
  return result;
}

struct CanonicalizedConfigurationABI {
  fabric::FabricSystemRootView system;
  std::vector<ConfigurationEncodingRelation> encodingRelations;
  std::vector<ProgrammingUnit> units;
  ConfigurationABIConstructionStatistics statistics;
};

void accumulateStatistics(ConfigurationABIConstructionStatistics &total,
                          const ConfigurationABIConstructionStatistics &part) {
  total.canonicalizationCount += part.canonicalizationCount;
  total.canonicalizationNanoseconds += part.canonicalizationNanoseconds;
  total.semanticValidationCacheHits += part.semanticValidationCacheHits;
  total.semanticValidationCacheMisses += part.semanticValidationCacheMisses;
  total.physicalSlotValidationCount += part.physicalSlotValidationCount;
  total.retainedCacheBytes += part.retainedCacheBytes;
  total.deterministicWork += part.deterministicWork;
  total.encodingRelationCount = part.encodingRelationCount;
  total.configurationFieldCount = part.configurationFieldCount;
  total.canonicalByteCount = part.canonicalByteCount;
}

bool sameEncodingRelation(const ConfigurationEncodingRelationDraft &lhs,
                          const ConfigurationEncodingRelationDraft &rhs) {
  return lhs.inactiveValue == rhs.inactiveValue &&
         sameSemanticEncoding(lhs.semanticEncoding, rhs.semanticEncoding);
}

bool encodingRelationLess(const ConfigurationEncodingRelationDraft &lhs,
                          const ConfigurationEncodingRelationDraft &rhs) {
  if (lhs.semanticEncoding.index() != rhs.semanticEncoding.index())
    return lhs.semanticEncoding.index() < rhs.semanticEncoding.index();
  if (const auto *left =
          std::get_if<DirectBitsEncoding>(&lhs.semanticEncoding)) {
    const auto &right = std::get<DirectBitsEncoding>(rhs.semanticEncoding);
    return std::tie(left->encodedBitCount, lhs.inactiveValue) <
           std::tie(right.encodedBitCount, rhs.inactiveValue);
  }
  const auto &left = std::get<FiniteCodebookEncoding>(lhs.semanticEncoding);
  const auto &right = std::get<FiniteCodebookEncoding>(rhs.semanticEncoding);
  if (left.encodedBitCount != right.encodedBitCount)
    return left.encodedBitCount < right.encodedBitCount;
  if (left.entries != right.entries)
    return std::lexicographical_compare(
        left.entries.begin(), left.entries.end(), right.entries.begin(),
        right.entries.end(), [](const auto &leftEntry, const auto &rightEntry) {
          return std::tie(leftEntry.semanticValue, leftEntry.physicalCode) <
                 std::tie(rightEntry.semanticValue, rightEntry.physicalCode);
        });
  return lhs.inactiveValue < rhs.inactiveValue;
}

llvm::Expected<CanonicalizedConfigurationABI>
canonicalizeDraft(ConfigurationABIDraft draft, const ArtifactStore &store) {
  const auto started = std::chrono::steady_clock::now();
  auto importedFabric =
      contextual(fabric::importEntireFabricRoot(draft.fabric, store),
                 "ConfigurationABI fabric_ref cannot be imported");
  if (!importedFabric)
    return importedFabric.takeError();
  auto system = contextual(fabric::requireSystemRoot(importedFabric->view()),
                           "ConfigurationABI requires an exact System root");
  if (!system)
    return system.takeError();

  struct IndexedRelation final {
    ConfigurationEncodingRelationId originalId = 0;
    ConfigurationEncodingRelationDraft relation;
  };
  std::vector<IndexedRelation> indexedRelations;
  indexedRelations.reserve(draft.encodingRelations.size());
  for (auto indexed : llvm::enumerate(draft.encodingRelations)) {
    if (llvm::Error error = canonicalizeEncoding(indexed.value()))
      return error;
    indexedRelations.push_back(
        {static_cast<ConfigurationEncodingRelationId>(indexed.index()),
         std::move(indexed.value())});
  }
  llvm::sort(indexedRelations, [](const auto &lhs, const auto &rhs) {
    return encodingRelationLess(lhs.relation, rhs.relation);
  });
  std::vector<ConfigurationEncodingRelationId> relationRemap(
      indexedRelations.size());
  std::vector<ConfigurationEncodingRelationDraft> uniqueRelations;
  uniqueRelations.reserve(indexedRelations.size());
  for (IndexedRelation &indexed : indexedRelations) {
    if (uniqueRelations.empty() ||
        !sameEncodingRelation(uniqueRelations.back(), indexed.relation))
      uniqueRelations.push_back(std::move(indexed.relation));
    relationRemap[static_cast<std::size_t>(indexed.originalId)] =
        static_cast<ConfigurationEncodingRelationId>(uniqueRelations.size() -
                                                     1);
  }
  std::vector<std::uint64_t> relationUseCounts(uniqueRelations.size(), 0);

  std::set<ByteVector> allOwners;
  std::set<ByteVector> allFields;
  std::map<ByteVector, ConfigurationEncodingRelationId> encodingByPhysicalField;
  mlir::MLIRContext semanticContext;
  std::set<SemanticValidationKey> semanticValidationCache;
  std::uint64_t semanticValidationCacheHits = 0;
  std::uint64_t semanticValidationCacheMisses = 0;
  std::uint64_t deterministicWork = uniqueRelations.size();
  std::vector<ProgrammingUnit> units;
  units.reserve(draft.programmingUnits.size());
  for (ProgrammingUnitDraft &unit : draft.programmingUnits) {
    if (unit.payloadBitCount == 0)
      return invalid("programming unit has zero payload_bit_count");
    if (!byteCountForBits(unit.payloadBitCount, "programming unit payload"))
      return invalid("programming unit payload is too large for this host");
    if (unit.exactFabricResourceClosure.empty())
      return invalid("programming unit resource closure is empty");
    if (unit.fields.empty())
      return invalid("programming unit has no configuration fields");

    using OwnerRow =
        std::pair<ByteVector, fabric::FabricPhysicalOccurrenceOwnerRef>;
    std::vector<OwnerRow> ownerRows;
    ownerRows.reserve(unit.exactFabricResourceClosure.size());
    std::set<ByteVector> unitOwners;
    for (fabric::FabricPhysicalOccurrenceOwnerRef &owner :
         unit.exactFabricResourceClosure) {
      auto resolved = system->resolvePhysicalOwner(owner);
      if (!resolved)
        return contextual(resolved.takeError(),
                          "resource closure owner does not resolve in Fabric");
      ByteVector key = referenceKey(owner);
      if (!unitOwners.insert(key).second)
        return invalid("programming unit resource closure is not a unique set");
      if (!allOwners.insert(key).second)
        return invalid("programming unit resource closures overlap");
      ownerRows.emplace_back(std::move(key), std::move(owner));
    }
    llvm::sort(ownerRows, [](const OwnerRow &lhs, const OwnerRow &rhs) {
      return lhs.first < rhs.first;
    });
    unit.exactFabricResourceClosure.clear();
    unit.exactFabricResourceClosure.reserve(ownerRows.size());
    for (OwnerRow &row : ownerRows)
      unit.exactFabricResourceClosure.push_back(std::move(row.second));

    std::vector<std::pair<std::uint64_t, std::uint64_t>> destinationRanges;
    using FieldRow = std::pair<ByteVector, ConfigurationFieldEncoding>;
    std::vector<FieldRow> fieldRows;
    fieldRows.reserve(unit.fields.size());
    for (ConfigurationFieldEncoding &field : unit.fields) {
      ++deterministicWork;
      if (field.encodingRelation >= relationRemap.size())
        return invalid("configuration field names an unknown encoding "
                       "relation");
      field.encodingRelation =
          relationRemap[static_cast<std::size_t>(field.encodingRelation)];
      ++relationUseCounts[static_cast<std::size_t>(field.encodingRelation)];
      const ConfigurationEncodingRelationDraft &encodingRelation =
          uniqueRelations[static_cast<std::size_t>(field.encodingRelation)];
      if (llvm::Error error =
              contextual(validatePhysicalSlot(*system, field.slot),
                         "configuration slot does not resolve in Fabric"))
        return std::move(error);
      ByteVector fieldKey = referenceKey(field.slot);
      if (!allFields.insert(fieldKey).second)
        return invalid("configuration slot belongs to more than one unit");
      const auto physicalField = fabric::configurationField(field.slot);
      auto physicalOwner = physicalOwnerForField(physicalField);
      if (!physicalOwner)
        return physicalOwner.takeError();
      if (unitOwners.find(referenceKey(*physicalOwner)) == unitOwners.end())
        return invalid(
            "configuration field owner is absent from its resource closure");
      const ByteVector physicalFieldKey = referenceKey(physicalField);
      auto [knownEncoding, inserted] = encodingByPhysicalField.emplace(
          physicalFieldKey, field.encodingRelation);
      if (!inserted && knownEncoding->second != field.encodingRelation)
        return invalid("configuration residencies of one physical field use "
                       "different semantic encodings");
      auto resolved = resolvePhysicalField(*system, physicalField);
      if (!resolved)
        return resolved.takeError();
      auto relationSource = fabric::semanticFieldRelationSourceIdentity(
          resolved->artifact, resolved->local);
      if (!relationSource)
        return relationSource.takeError();
      SemanticValidationKey validationKey{
          resolved->artifact.identity().bytes(), configurationAbiSchema.version,
          semanticValidationAlgorithmVersion,
          ByteVector(relationSource->bytes().begin(),
                     relationSource->bytes().end()),
          field.encodingRelation};
      auto [cached, cacheInserted] =
          semanticValidationCache.insert(std::move(validationKey));
      if (cacheInserted) {
        ++semanticValidationCacheMisses;
        ++deterministicWork;
        if (llvm::Error error = validateFieldEncoding(
                *resolved, encodingRelation, semanticContext))
          return error;
      } else {
        (void)cached;
        ++semanticValidationCacheHits;
      }
      if (llvm::Error error =
              canonicalizeSlices(field, encodingRelation.encodedBitCount(),
                                 unit.payloadBitCount, destinationRanges))
        return error;
      fieldRows.emplace_back(std::move(fieldKey), std::move(field));
    }
    llvm::sort(fieldRows, [](const FieldRow &lhs, const FieldRow &rhs) {
      return lhs.first < rhs.first;
    });
    unit.fields.clear();
    unit.fields.reserve(fieldRows.size());
    for (FieldRow &row : fieldRows)
      unit.fields.push_back(std::move(row.second));
    llvm::sort(destinationRanges);
    for (std::size_t index = 1; index < destinationRanges.size(); ++index)
      if (destinationRanges[index].first < destinationRanges[index - 1].second)
        return invalid("destination bits overlap");
    units.push_back(
        ProgrammingUnit{0, std::move(unit.exactFabricResourceClosure),
                        unit.payloadBitCount, std::move(unit.fields)});
  }

  auto expectedSlots = expectedPhysicalSlots(*system);
  if (!expectedSlots)
    return expectedSlots.takeError();
  if (allFields != *expectedSlots) {
    const auto missing =
        llvm::find_if(*expectedSlots, [&](const ByteVector &slot) {
          return !allFields.count(slot);
        });
    const auto unexpected =
        llvm::find_if(allFields, [&](const ByteVector &slot) {
          return !expectedSlots->count(slot);
        });
    std::string detail =
        "programming units do not cover every Fabric configuration slot "
        "(actual=" +
        std::to_string(allFields.size()) +
        ", expected=" + std::to_string(expectedSlots->size());
    if (missing != expectedSlots->end()) {
      detail += ", first_missing=" + formatArtifactLocalPayloadHex(*missing);
      auto decoded =
          fabric::decodeFabricRef<fabric::FabricPhysicalConfigurationSlotRef>(
              *missing);
      if (decoded) {
        const auto &local = fabric::configurationSlot(*decoded);
        detail +=
            ", missing_owner_kind=" +
            std::to_string(
                static_cast<unsigned>(local.field.owner.catalog().kind())) +
            ", missing_field_ordinal=" + std::to_string(local.field.ordinal);
        if (decoded->kind() == fabric::FabricPhysicalConfigurationSlotKind::
                                   SpatialCoreInternalSlot) {
          const auto &internal =
              std::get<fabric::SpatialCoreInternalConfigurationSlotRef>(
                  decoded->payload());
          detail += ", missing_spatial_core=" +
                    std::to_string(internal.spatialCore.core.id());
        }
        if (local.field.owner.catalog().kind() ==
            fabric::FabricInventoryOwnerKind::FuOccurrenceNode) {
          const auto &node = std::get<fabric::FabricFuOccurrenceNodeRef>(
              local.field.owner.catalog().payload);
          detail += ", missing_fu=" + std::to_string(node.fu.id()) +
                    ", missing_node_kind=" +
                    std::to_string(static_cast<unsigned>(node.node)) +
                    ", missing_node_ordinal=" + std::to_string(node.ordinal);
        }
        if (const auto *context =
                std::get_if<fabric::InstructionContextRef>(&local.residency))
          detail += ", missing_instruction_context=" +
                    std::to_string(context->ordinal);
        else
          detail += ", missing_residency=static";
      } else {
        llvm::consumeError(decoded.takeError());
      }
    }
    if (unexpected != allFields.end())
      detail +=
          ", first_unexpected=" + formatArtifactLocalPayloadHex(*unexpected);
    detail += ")";
    return invalid(detail);
  }

  using UnitRow = std::pair<ClosureKey, ProgrammingUnit>;
  std::vector<UnitRow> unitRows;
  unitRows.reserve(units.size());
  for (ProgrammingUnit &unit : units)
    unitRows.emplace_back(closureKey(unit), std::move(unit));
  llvm::sort(unitRows, [](const UnitRow &lhs, const UnitRow &rhs) {
    return lhs.first < rhs.first;
  });
  units.clear();
  units.reserve(unitRows.size());
  for (std::size_t index = 0; index < unitRows.size(); ++index) {
    if (index != 0 && unitRows[index - 1].first == unitRows[index].first)
      return invalid("duplicate programming unit resource closure");
    unitRows[index].second.id = static_cast<ProgrammingUnitId>(index);
    units.push_back(std::move(unitRows[index].second));
  }
  if (llvm::is_contained(relationUseCounts, std::uint64_t(0)))
    return invalid("configuration encoding relation has no field use");
  std::vector<ConfigurationEncodingRelation> finalizedRelations;
  finalizedRelations.reserve(uniqueRelations.size());
  for (auto indexed : llvm::enumerate(uniqueRelations))
    finalizedRelations.push_back(ConfigurationEncodingRelation{
        static_cast<ConfigurationEncodingRelationId>(indexed.index()),
        std::move(indexed.value().semanticEncoding),
        std::move(indexed.value().inactiveValue)});
  std::uint64_t retainedCacheBytes = 0;
  for (const SemanticValidationKey &key : semanticValidationCache)
    retainedCacheBytes += sizeof(key) + key.localField.size();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started);
  ConfigurationABIConstructionStatistics statistics;
  statistics.canonicalizationCount = 1;
  statistics.canonicalizationNanoseconds = elapsed.count();
  statistics.semanticValidationCacheHits = semanticValidationCacheHits;
  statistics.semanticValidationCacheMisses = semanticValidationCacheMisses;
  statistics.physicalSlotValidationCount = allFields.size();
  statistics.retainedCacheBytes = retainedCacheBytes;
  statistics.deterministicWork = deterministicWork;
  statistics.encodingRelationCount = finalizedRelations.size();
  statistics.configurationFieldCount = allFields.size();
  return CanonicalizedConfigurationABI{std::move(*system),
                                       std::move(finalizedRelations),
                                       std::move(units), statistics};
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.attribute("schema", reference.schemaIdentity);
  json.attribute("schema_version",
                 formatSchemaVersion(reference.schemaVersion));
  json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
}

void writeSemanticEncoding(llvm::json::OStream &json,
                           const SemanticFieldEncoding &encoding) {
  std::visit(
      [&](const auto &value) {
        using Encoding = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Encoding, DirectBitsEncoding>) {
          json.attribute("kind", "direct_bits");
          json.attribute("encoded_bit_count", value.encodedBitCount);
        } else {
          json.attribute("kind", "finite_codebook");
          json.attribute("encoded_bit_count", value.encodedBitCount);
          json.attributeArray("entries", [&] {
            for (const FiniteCodebookEntry &entry : value.entries) {
              json.object([&] {
                json.attribute("semantic_value", formatArtifactLocalPayloadHex(
                                                     entry.semanticValue));
                json.attribute("physical_code", formatArtifactLocalPayloadHex(
                                                    entry.physicalCode));
              });
            }
          });
        }
      },
      encoding);
}

std::string serializeConfigurationABI(
    const ArtifactRootReference &fabricReference,
    llvm::ArrayRef<ConfigurationEncodingRelation> encodingRelations,
    llvm::ArrayRef<ProgrammingUnit> programmingUnits) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", configurationAbiSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(configurationAbiSchema.version));
    json.attributeObject("fabric_ref",
                         [&] { writeRootReference(json, fabricReference); });
    json.attributeArray("encoding_relations", [&] {
      for (const ConfigurationEncodingRelation &relation : encodingRelations) {
        json.object([&] {
          json.attribute("relation_id", relation.id);
          json.attributeObject("semantic_encoding", [&] {
            writeSemanticEncoding(json, relation.semanticEncoding);
          });
          json.attribute("inactive_value",
                         formatArtifactLocalPayloadHex(relation.inactiveValue));
        });
      }
    });
    json.attributeArray("programming_units", [&] {
      for (const ProgrammingUnit &unit : programmingUnits) {
        json.object([&] {
          json.attribute("unit_id", unit.id);
          json.attributeArray("exact_fabric_resource_closure", [&] {
            for (const fabric::FabricPhysicalOccurrenceOwnerRef &owner :
                 unit.exactFabricResourceClosure)
              json.value(formatArtifactLocalPayloadHex(referenceKey(owner)));
          });
          json.attributeObject("programming_model", [&] {
            json.attribute("kind", "complete_image_atomic");
            json.attribute("payload_bit_count", unit.payloadBitCount);
          });
          json.attributeArray("fields", [&] {
            for (const ConfigurationFieldEncoding &field : unit.fields) {
              json.object([&] {
                json.attribute(
                    "fabric_config_slot_ref",
                    formatArtifactLocalPayloadHex(referenceKey(field.slot)));
                json.attribute("encoding_relation_id", field.encodingRelation);
                json.attributeArray("destination_slices", [&] {
                  for (const DestinationSlice &slice :
                       field.destinationSlices) {
                    json.object([&] {
                      json.attribute("source_bit_offset",
                                     slice.sourceBitOffset);
                      json.attribute("destination_bit_offset",
                                     slice.destinationBitOffset);
                      json.attribute("bit_count", slice.bitCount);
                    });
                  }
                });
              });
            }
          });
        });
      }
    });
  });
  return output.str().str();
}

const ConfigurationFieldEncoding *
findFieldInUnit(const ProgrammingUnit &unit, llvm::ArrayRef<std::uint8_t> key) {
  const auto found = std::lower_bound(
      unit.fields.begin(), unit.fields.end(), key,
      [](const ConfigurationFieldEncoding &field,
         llvm::ArrayRef<std::uint8_t> selected) {
        const ByteVector candidate = referenceKey(field.slot);
        return std::lexicographical_compare(
            candidate.begin(), candidate.end(), selected.begin(),
            selected.end());
      });
  if (found == unit.fields.end())
    return nullptr;
  return llvm::ArrayRef<std::uint8_t>(referenceKey(found->slot)).equals(key)
             ? &*found
             : nullptr;
}

llvm::Expected<ByteVector>
encodeField(const ConfigurationEncodingRelation &relation,
            llvm::ArrayRef<std::uint8_t> semanticValue) {
  if (const auto *direct =
          std::get_if<DirectBitsEncoding>(&relation.semanticEncoding)) {
    if (llvm::Error error = validateBitVector(
            semanticValue, direct->encodedBitCount, "DirectBits value"))
      return std::move(error);
    return ByteVector(semanticValue.begin(), semanticValue.end());
  }
  const auto &codebook =
      std::get<FiniteCodebookEncoding>(relation.semanticEncoding);
  const auto entry = llvm::find_if(
      codebook.entries, [&](const FiniteCodebookEntry &candidate) {
        return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(semanticValue);
      });
  if (entry == codebook.entries.end())
    return invalid("finite codebook has no entry for semantic value");
  return entry->physicalCode;
}

std::uint64_t retainedConfigurationABIBytes(
    const CanonicalSemanticBytes &canonicalBytes,
    const ConfigurationABI &abi) {
  std::uint64_t bytes = sizeof(ConfigurationABI) + canonicalBytes.bytes().size();
  const auto add = [&](std::uint64_t amount) {
    if (amount > std::numeric_limits<std::uint64_t>::max() - bytes)
      bytes = std::numeric_limits<std::uint64_t>::max();
    else
      bytes += amount;
  };
  add(abi.encodingRelations().size() *
      sizeof(ConfigurationEncodingRelation));
  for (const ConfigurationEncodingRelation &relation :
       abi.encodingRelations()) {
    add(relation.inactiveValue.capacity());
    const auto *codebook =
        std::get_if<FiniteCodebookEncoding>(&relation.semanticEncoding);
    if (!codebook)
      continue;
    add(codebook->entries.capacity() * sizeof(FiniteCodebookEntry));
    for (const FiniteCodebookEntry &entry : codebook->entries) {
      add(entry.semanticValue.capacity());
      add(entry.physicalCode.capacity());
    }
  }
  add(abi.programmingUnits().size() * sizeof(ProgrammingUnit));
  for (const ProgrammingUnit &unit : abi.programmingUnits()) {
    add(unit.exactFabricResourceClosure.capacity() *
        sizeof(fabric::FabricPhysicalOccurrenceOwnerRef));
    add(unit.fields.capacity() * sizeof(ConfigurationFieldEncoding));
    for (const ConfigurationFieldEncoding &field : unit.fields)
      add(field.destinationSlices.capacity() * sizeof(DestinationSlice));
  }
  return bytes;
}

} // namespace

namespace detail {

constexpr std::uint64_t configurationABIImportAlgorithmVersion = 1;

struct ConfigurationABIImportSessionKey final {
  ArtifactRootReference reference;
  std::uint64_t algorithmVersion = configurationABIImportAlgorithmVersion;
};

bool operator==(const ConfigurationABIImportSessionKey &lhs,
                const ConfigurationABIImportSessionKey &rhs) {
  return lhs.reference == rhs.reference &&
         lhs.algorithmVersion == rhs.algorithmVersion;
}

struct ConfigurationABIImportSessionEntry final {
  ConfigurationABIImportSessionKey key;
  CanonicalSemanticBytes canonicalBytes;
  std::shared_ptr<const ConfigurationABI> abi;
  ConfigurationABIConstructionStatistics constructionStatistics;
  std::uint64_t retainedBytes = 0;
};

class ConfigurationABIImportSessionState final {
public:
  void recordRequest() {
    add(statistics_.importRequests, 1);
  }

  void recordRead(std::uint64_t byteCount) {
    add(statistics_.bytesRead, byteCount);
    add(statistics_.bytesCopied, byteCount);
  }

  std::shared_ptr<const ConfigurationABIImportSessionEntry>
  find(const ArtifactRootReference &reference) {
    add(statistics_.deterministicWork, 1);
    const ConfigurationABIImportSessionKey key{reference};
    const auto found = llvm::find_if(
        entries_, [&](const auto &entry) { return entry->key == key; });
    if (found == entries_.end()) {
      add(statistics_.cacheMisses, 1);
      return {};
    }
    add(statistics_.cacheHits, 1);
    return *found;
  }

  std::shared_ptr<const ConfigurationABIImportSessionEntry> insert(
      const ArtifactRootReference &reference,
      CanonicalSemanticBytes canonicalBytes,
      std::shared_ptr<const ConfigurationABI> abi,
      const ConfigurationABIConstructionStatistics &constructionStatistics,
      std::uint64_t retainedBytes, std::uint64_t constructionNanoseconds) {
    const ConfigurationABIImportSessionKey key{reference};
    const auto found = llvm::find_if(
        entries_, [&](const auto &entry) { return entry->key == key; });
    if (found != entries_.end())
      return *found;
    auto entry = std::make_shared<const ConfigurationABIImportSessionEntry>(
        ConfigurationABIImportSessionEntry{
            key, std::move(canonicalBytes), std::move(abi),
            constructionStatistics, retainedBytes});
    entries_.push_back(entry);
    add(statistics_.uniqueConstructions, 1);
    add(statistics_.constructionNanoseconds, constructionNanoseconds);
    add(statistics_.deterministicWork,
        constructionStatistics.deterministicWork);
    add(statistics_.retainedBytes, retainedBytes);
    statistics_.entryCount = entries_.size();
    return entry;
  }

  ConfigurationABIImportSessionStatistics statistics() const {
    return statistics_;
  }

private:
  static void add(std::uint64_t &destination, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - destination)
      destination = std::numeric_limits<std::uint64_t>::max();
    else
      destination += value;
  }

  std::list<std::shared_ptr<const ConfigurationABIImportSessionEntry>> entries_;
  ConfigurationABIImportSessionStatistics statistics_;
};

} // namespace detail

ConfigurationABIImportSession::ConfigurationABIImportSession(
    ConfigurationABIImportSessionMode mode)
    : previous_(currentConfigurationABIImportSession) {
  if (mode == ConfigurationABIImportSessionMode::ReuseEnclosing && previous_) {
    active_ = previous_;
  } else {
    owned_ = std::make_unique<detail::ConfigurationABIImportSessionState>();
    active_ = owned_.get();
  }
  currentConfigurationABIImportSession = active_;
}

ConfigurationABIImportSession::~ConfigurationABIImportSession() {
  currentConfigurationABIImportSession = previous_;
}

ConfigurationABIImportSessionStatistics
ConfigurationABIImportSession::statistics() const {
  return active_ ? active_->statistics()
                 : ConfigurationABIImportSessionStatistics{};
}

ProgrammingUnitOccurrenceScope
deriveProgrammingUnitOccurrenceScope(const ProgrammingUnit &unit) {
  ProgrammingUnitOccurrenceScope result;
  for (const fabric::FabricPhysicalOccurrenceOwnerRef &owner :
       unit.exactFabricResourceClosure) {
    if (owner.kind() !=
        fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal) {
      result.includesDirectSystemResources = true;
      continue;
    }
    const auto &spatialCore =
        std::get<fabric::SpatialCoreInternalOccurrenceRef>(owner.payload())
            .spatialCore;
    if (!llvm::is_contained(result.spatialCores, spatialCore))
      result.spatialCores.push_back(spatialCore);
  }
  llvm::sort(result.spatialCores, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  return result;
}

const ConfigurationEncodingRelation *ConfigurationABI::findEncodingRelation(
    ConfigurationEncodingRelationId id) const {
  if (id >= encodingRelations_.size())
    return nullptr;
  const ConfigurationEncodingRelation &relation =
      encodingRelations_[static_cast<std::size_t>(id)];
  return relation.id == id ? &relation : nullptr;
}

const ProgrammingUnit *
ConfigurationABI::findProgrammingUnit(ProgrammingUnitId id) const {
  if (id >= programmingUnits_.size())
    return nullptr;
  const ProgrammingUnit &unit = programmingUnits_[static_cast<std::size_t>(id)];
  return unit.id == id ? &unit : nullptr;
}

const ConfigurationFieldEncoding *ConfigurationABI::findField(
    const fabric::FabricPhysicalConfigurationSlotRef &slot) const {
  const ByteVector key = referenceKey(slot);
  for (const ProgrammingUnit &unit : programmingUnits_)
    if (const ConfigurationFieldEncoding *encoding = findFieldInUnit(unit, key))
      return encoding;
  return nullptr;
}

const ConfigurationFieldEncoding *ConfigurationABI::findField(
    ProgrammingUnitId unitId,
    const fabric::FabricPhysicalConfigurationSlotRef &slot) const {
  const ProgrammingUnit *unit = findProgrammingUnit(unitId);
  return unit ? findFieldInUnit(*unit, referenceKey(slot)) : nullptr;
}

const ConfigurationFieldEncoding *ConfigurationABI::findOperationField(
    const fabric::FabricPhysicalOccurrenceOwnerRef &operation,
    fabric::FabricOrdinal fieldOrdinal) const {
  if (operation.kind() !=
      fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
    return nullptr;
  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(operation.payload());
  if (internal.target.kind() != fabric::FabricModulePhysicalTargetKind::Owner)
    return nullptr;
  const auto &owner =
      std::get<fabric::FabricModulePhysicalOwnerRef>(internal.target.payload());
  if (owner.kind() != fabric::FabricModulePhysicalOwnerKind::FuOccurrenceNode)
    return nullptr;
  const auto localOccurrence =
      std::get<fabric::FabricFuOccurrenceNodeRef>(owner.payload());
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(localOccurrence)),
      fieldOrdinal};
  auto physical = qualifyModuleField(internal.spatialCore, field);
  if (!physical) {
    llvm::consumeError(physical.takeError());
    return nullptr;
  }
  const ByteVector key = referenceKey(*physical);
  for (const ProgrammingUnit &unit : programmingUnits_)
    for (const ConfigurationFieldEncoding &candidate : unit.fields)
      if (referenceKey(fabric::configurationField(candidate.slot)) == key)
        return &candidate;
  return nullptr;
}

const ConfigurationEncodingRelation *
ConfigurationABI::findOperationEncodingRelation(
    const fabric::FabricPhysicalOccurrenceOwnerRef &operation,
    fabric::FabricOrdinal fieldOrdinal) const {
  const ConfigurationFieldEncoding *field =
      findOperationField(operation, fieldOrdinal);
  return field ? findEncodingRelation(*field) : nullptr;
}

llvm::Expected<std::vector<std::uint8_t>> ConfigurationABI::encode(
    ProgrammingUnitId id,
    llvm::ArrayRef<SemanticConfigurationValue> values) const {
  const ProgrammingUnit *unit = findProgrammingUnit(id);
  if (!unit)
    return invalid("unknown programming unit");
  auto payloadByteCount =
      byteCountForBits(unit->payloadBitCount, "programming unit payload");
  if (!payloadByteCount)
    return payloadByteCount.takeError();
  std::vector<std::uint8_t> payload(*payloadByteCount, 0);

  std::map<std::size_t, llvm::ArrayRef<std::uint8_t>> selected;
  for (const SemanticConfigurationValue &value : values) {
    ByteVector key = referenceKey(value.slot);
    const ConfigurationFieldEncoding *field = findFieldInUnit(*unit, key);
    if (!field)
      return invalid(
          "semantic value names a field outside the programming unit");
    const std::size_t ordinal =
        static_cast<std::size_t>(field - unit->fields.data());
    if (!selected.emplace(ordinal, value.value).second)
      return invalid("semantic value names a configuration field twice");
  }

  std::vector<ByteVector> inactiveEncodings;
  inactiveEncodings.reserve(encodingRelations_.size());
  for (const ConfigurationEncodingRelation &relation : encodingRelations_) {
    auto encoded = encodeField(relation, relation.inactiveValue);
    if (!encoded)
      return encoded.takeError();
    inactiveEncodings.push_back(std::move(*encoded));
  }

  for (const auto indexed : llvm::enumerate(unit->fields)) {
    const ConfigurationFieldEncoding &field = indexed.value();
    const ConfigurationEncodingRelation *relation = findEncodingRelation(field);
    if (!relation)
      return invalid("configuration field names an unknown encoding relation");
    const auto found = selected.find(indexed.index());
    ByteVector selectedEncoding;
    llvm::ArrayRef<std::uint8_t> encoded;
    if (found == selected.end()) {
      encoded = inactiveEncodings[static_cast<std::size_t>(relation->id)];
      if (llvm::all_of(encoded, [](std::uint8_t byte) { return byte == 0; }))
        continue;
    } else {
      if (llvm::Error error = validateSemanticValue(
              system_, fabric::configurationField(field.slot), found->second))
        return std::move(error);
      auto selectedValue = encodeField(*relation, found->second);
      if (!selectedValue)
        return selectedValue.takeError();
      selectedEncoding = std::move(*selectedValue);
      encoded = selectedEncoding;
    }
    for (const DestinationSlice &slice : field.destinationSlices)
      for (std::uint64_t index = 0; index < slice.bitCount; ++index)
        setBit(payload, slice.destinationBitOffset + index,
               bit(encoded, slice.sourceBitOffset + index));
  }
  return payload;
}

llvm::Expected<std::vector<SemanticConfigurationValue>>
ConfigurationABI::decode(ProgrammingUnitId id,
                         llvm::ArrayRef<std::uint8_t> payload) const {
  const ProgrammingUnit *unit = findProgrammingUnit(id);
  if (!unit)
    return invalid("unknown programming unit");
  if (llvm::Error error =
          validateBitVector(payload, unit->payloadBitCount, "payload"))
    return std::move(error);

  auto occupiedByteCount =
      byteCountForBits(unit->payloadBitCount, "programming unit payload");
  if (!occupiedByteCount)
    return occupiedByteCount.takeError();
  std::vector<std::uint8_t> occupied(*occupiedByteCount, 0);
  std::vector<SemanticConfigurationValue> values;
  values.reserve(unit->fields.size());
  for (const ConfigurationFieldEncoding &field : unit->fields) {
    const ConfigurationEncodingRelation *relation = findEncodingRelation(field);
    if (!relation)
      return invalid("configuration field names an unknown encoding relation");
    const std::uint64_t fieldBitCount = relation->encodedBitCount();
    auto fieldByteCount =
        byteCountForBits(fieldBitCount, "configuration field");
    if (!fieldByteCount)
      return fieldByteCount.takeError();
    ByteVector physical(*fieldByteCount, 0);
    for (const DestinationSlice &slice : field.destinationSlices) {
      for (std::uint64_t index = 0; index < slice.bitCount; ++index) {
        setBit(physical, slice.sourceBitOffset + index,
               bit(payload, slice.destinationBitOffset + index));
        setBit(occupied, slice.destinationBitOffset + index, true);
      }
    }

    ByteVector semantic;
    if (std::holds_alternative<DirectBitsEncoding>(
            relation->semanticEncoding)) {
      semantic = std::move(physical);
    } else {
      const auto &codebook =
          std::get<FiniteCodebookEncoding>(relation->semanticEncoding);
      const auto entry = llvm::find_if(
          codebook.entries, [&](const FiniteCodebookEntry &candidate) {
            return candidate.physicalCode == physical;
          });
      if (entry == codebook.entries.end())
        return invalid("finite codebook cannot decode physical code");
      semantic = entry->semanticValue;
    }
    if (llvm::Error error = validateSemanticValue(
            system_, fabric::configurationField(field.slot), semantic))
      return std::move(error);
    values.push_back(
        SemanticConfigurationValue{field.slot, std::move(semantic)});
  }

  for (std::uint64_t index = 0; index < unit->payloadBitCount; ++index)
    if (!bit(occupied, index) && bit(payload, index))
      return invalid("reserved bit is nonzero");
  return values;
}

llvm::Expected<FinalizedConfigurationABI>
finalizeConfigurationABI(ConfigurationABIDraft draft,
                         const ArtifactStore &store) {
  ArtifactRootReference fabricReference = draft.fabric;
  auto units = canonicalizeDraft(std::move(draft), store);
  if (!units)
    return units.takeError();
  const std::string json = serializeConfigurationABI(
      fabricReference, units->encodingRelations, units->units);
  units->statistics.canonicalByteCount = json.size();
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));

  auto reparsedDraft = parseConfigurationABI(bytes.bytes());
  if (!reparsedDraft)
    return reparsedDraft.takeError();
  ArtifactRootReference reparsedFabric = reparsedDraft->fabric;
  auto reparsedUnits = canonicalizeDraft(std::move(*reparsedDraft), store);
  if (!reparsedUnits)
    return reparsedUnits.takeError();
  reparsedUnits->statistics.canonicalByteCount = json.size();
  if (serializeConfigurationABI(reparsedFabric,
                                reparsedUnits->encodingRelations,
                                reparsedUnits->units) != json)
    return invalid("canonical JSON does not independently round-trip");

  auto identity = store.put(configurationAbiSchema, bytes);
  if (!identity)
    return identity.takeError();
  auto imported =
      importConfigurationABI({configurationAbiSchema.identity.str(),
                              configurationAbiSchema.version, *identity},
                             store);
  if (!imported)
    return imported.takeError();
  ConfigurationABIConstructionStatistics aggregate = units->statistics;
  accumulateStatistics(aggregate, reparsedUnits->statistics);
  accumulateStatistics(aggregate, imported->constructionStatistics_);
  aggregate.canonicalByteCount = json.size();
  imported->constructionStatistics_ = aggregate;
  return std::move(*imported);
}

llvm::Expected<FinalizedConfigurationABI>
importConfigurationABI(const ArtifactRootReference &reference,
                       const ArtifactStore &store) {
  if (reference.schemaIdentity != configurationAbiSchema.identity ||
      reference.schemaVersion != configurationAbiSchema.version)
    return invalid("reference is not loom.configuration_abi 4.0");
  if (currentConfigurationABIImportSession) {
    currentConfigurationABIImportSession->recordRequest();
    if (auto cached =
            currentConfigurationABIImportSession->find(reference))
      return FinalizedConfigurationABI(
          reference, cached->canonicalBytes, cached->abi,
          cached->constructionStatistics);
  }
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  if (currentConfigurationABIImportSession)
    currentConfigurationABIImportSession->recordRead(bytes->bytes().size());
  const auto started = std::chrono::steady_clock::now();
  auto draft = parseConfigurationABI(bytes->bytes());
  if (!draft)
    return draft.takeError();
  ArtifactRootReference fabricReference = draft->fabric;
  auto units = canonicalizeDraft(std::move(*draft), store);
  if (!units)
    return units.takeError();
  units->statistics.canonicalByteCount = bytes->bytes().size();
  const std::string rewritten = serializeConfigurationABI(
      fabricReference, units->encodingRelations, units->units);
  llvm::StringRef stored(reinterpret_cast<const char *>(bytes->bytes().data()),
                         bytes->bytes().size());
  if (stored != rewritten)
    return invalid("stored ConfigurationABI payload is not canonical");
  std::shared_ptr<const ConfigurationABI> abi(new ConfigurationABI(
      fabricReference, std::move(units->encodingRelations),
      std::move(units->units), std::move(units->system)));
  if (currentConfigurationABIImportSession) {
    const std::uint64_t retainedBytes =
        retainedConfigurationABIBytes(*bytes, *abi);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    auto cached = currentConfigurationABIImportSession->insert(
        reference, *bytes, std::move(abi), units->statistics, retainedBytes,
        static_cast<std::uint64_t>(elapsed.count()));
    return FinalizedConfigurationABI(
        reference, cached->canonicalBytes, cached->abi,
        cached->constructionStatistics);
  }
  return FinalizedConfigurationABI(reference, std::move(*bytes), std::move(abi),
                                   units->statistics);
}

} // namespace loom::hardware

#include "Hardware/Configuration/ConfigurationABI.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

llvm::Expected<ConfigurationFieldEncoding>
parseField(const llvm::json::Value &value) {
  constexpr llvm::StringLiteral context = "configuration field";
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          rejectUnknownFields(*object, context,
                              {"fabric_config_slot_ref", "semantic_encoding",
                               "destination_slices", "inactive_value"}))
    return std::move(error);
  auto slotText = requireString(*object, "fabric_config_slot_ref", context);
  if (!slotText)
    return slotText.takeError();
  auto slot = parseReference<fabric::FabricPhysicalConfigurationSlotRef>(
      *slotText, "fabric_config_slot_ref");
  if (!slot)
    return slot.takeError();
  auto encodingObject = requireObject(*object, "semantic_encoding", context);
  if (!encodingObject)
    return encodingObject.takeError();
  auto encoding = parseSemanticEncoding(**encodingObject);
  if (!encoding)
    return encoding.takeError();
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
  auto inactive = requireString(*object, "inactive_value", context);
  if (!inactive)
    return inactive.takeError();
  auto inactiveBytes =
      contextual(parseArtifactLocalPayloadHex(*inactive), context);
  if (!inactiveBytes)
    return inactiveBytes.takeError();
  return ConfigurationFieldEncoding{std::move(*slot), std::move(*encoding),
                                    std::move(parsedSlices),
                                    std::move(*inactiveBytes)};
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
  if (llvm::Error error = rejectUnknownFields(
          *root, "ConfigurationABI",
          {"schema", "schema_version", "fabric_ref", "programming_units"}))
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
                               std::move(parsedUnits)};
}

llvm::Error canonicalizeEncoding(ConfigurationFieldEncoding &field) {
  const std::uint64_t bitCount = field.encodedBitCount();
  if (bitCount == 0)
    return invalid("semantic encoding has zero encoded_bit_count");
  if (auto *direct = std::get_if<DirectBitsEncoding>(&field.semanticEncoding)) {
    (void)direct;
    return validateBitVector(field.inactiveValue, bitCount,
                             "DirectBits inactive value");
  }

  auto &codebook = std::get<FiniteCodebookEncoding>(field.semanticEncoding);
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
        return entry.semanticValue == field.inactiveValue;
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

llvm::Error validateFieldEncoding(const fabric::FabricSystemRootView &system,
                                  const ConfigurationFieldEncoding &field) {
  const auto physicalField = fabric::configurationField(field.slot);
  auto resolved = resolvePhysicalField(system, physicalField);
  if (!resolved)
    return resolved.takeError();
  mlir::MLIRContext context;
  auto relation =
      resolved->artifact.semanticFieldRelation(resolved->local, context);
  if (!relation)
    return relation.takeError();
  switch (relation->kind()) {
  case fabric::FabricSemanticFieldRelationKind::None:
    return invalid("field is present for a fixed Fabric resource");
  case fabric::FabricSemanticFieldRelationKind::Finite: {
    const auto *codebook =
        std::get_if<FiniteCodebookEncoding>(&field.semanticEncoding);
    if (!codebook)
      return invalid("finite Fabric field requires a finite codebook");
    auto expected = finiteSemanticDomain(*resolved, context);
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

    const auto &owner = resolved->local.owner.catalog();
    if (owner.kind() == fabric::FabricInventoryOwnerKind::PeOccurrence) {
      auto schema = resolved->artifact.spatialPeConfigurationSchema(
          std::get<fabric::FabricPeOccurrenceRef>(owner.payload));
      if (!schema)
        return schema.takeError();
      const auto descriptor =
          llvm::find_if(schema->fields(), [&](const auto &candidate) {
            return candidate.reference == resolved->local;
          });
      if (descriptor == schema->fields().end())
        return invalid("PE configuration field is absent from its schema");
      if (descriptor->kind ==
          fabric::FabricPeConfigurationFieldKind::Activation) {
        auto disabled =
            schema->encode(resolved->local, fabric::FabricPeDisabled{});
        if (!disabled)
          return disabled.takeError();
        if (!disabled->bytes().equals(field.inactiveValue))
          return invalid("PE activation inactive value is not Disabled");
      }
    }
    return llvm::Error::success();
  }
  case fabric::FabricSemanticFieldRelationKind::Direct: {
    const auto *direct =
        std::get_if<DirectBitsEncoding>(&field.semanticEncoding);
    if (!direct)
      return invalid("direct Fabric field requires DirectBits");
    if (!relation->directEncodedBitCount() ||
        direct->encodedBitCount != *relation->directEncodedBitCount())
      return invalid("DirectBits width does not equal its Fabric relation");
    return relation->validateSemanticValue(field.inactiveValue);
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
    ConfigurationFieldEncoding &field, std::uint64_t payloadBitCount,
    std::vector<std::pair<std::uint64_t, std::uint64_t>> &destinationRanges) {
  const std::uint64_t sourceLimit = field.encodedBitCount();
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
  std::vector<ProgrammingUnit> units;
};

llvm::Expected<CanonicalizedConfigurationABI>
canonicalizeDraft(ConfigurationABIDraft draft, const ArtifactStore &store) {
  auto importedFabric =
      contextual(fabric::importEntireFabricRoot(draft.fabric, store),
                 "ConfigurationABI fabric_ref cannot be imported");
  if (!importedFabric)
    return importedFabric.takeError();
  auto system = contextual(fabric::requireSystemRoot(importedFabric->view()),
                           "ConfigurationABI requires an exact System root");
  if (!system)
    return system.takeError();

  std::set<ByteVector> allOwners;
  std::set<ByteVector> allFields;
  std::map<ByteVector, SemanticFieldEncoding> encodingByPhysicalField;
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

    llvm::sort(unit.exactFabricResourceClosure,
               [](const fabric::FabricPhysicalOccurrenceOwnerRef &lhs,
                  const fabric::FabricPhysicalOccurrenceOwnerRef &rhs) {
                 return referenceKey(lhs) < referenceKey(rhs);
               });
    std::set<ByteVector> unitOwners;
    for (const fabric::FabricPhysicalOccurrenceOwnerRef &owner :
         unit.exactFabricResourceClosure) {
      auto resolved = system->resolvePhysicalOwner(owner);
      if (!resolved)
        return contextual(resolved.takeError(),
                          "resource closure owner does not resolve in Fabric");
      ByteVector key = referenceKey(owner);
      if (!unitOwners.insert(key).second)
        return invalid("programming unit resource closure is not a unique set");
      if (!allOwners.insert(std::move(key)).second)
        return invalid("programming unit resource closures overlap");
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> destinationRanges;
    for (ConfigurationFieldEncoding &field : unit.fields) {
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
      if (llvm::Error error = canonicalizeEncoding(field))
        return error;
      const ByteVector physicalFieldKey = referenceKey(physicalField);
      auto [knownEncoding, inserted] = encodingByPhysicalField.emplace(
          physicalFieldKey, field.semanticEncoding);
      if (!inserted &&
          !sameSemanticEncoding(knownEncoding->second,
                                field.semanticEncoding))
        return invalid("configuration residencies of one physical field use "
                       "different semantic encodings");
      if (llvm::Error error = validateFieldEncoding(*system, field))
        return error;
      if (llvm::Error error = canonicalizeSlices(field, unit.payloadBitCount,
                                                 destinationRanges))
        return error;
    }
    llvm::sort(unit.fields, [](const ConfigurationFieldEncoding &lhs,
                               const ConfigurationFieldEncoding &rhs) {
      return referenceKey(lhs.slot) < referenceKey(rhs.slot);
    });
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

  llvm::sort(units, [](const ProgrammingUnit &lhs, const ProgrammingUnit &rhs) {
    return closureKey(lhs) < closureKey(rhs);
  });
  for (std::size_t index = 0; index < units.size(); ++index) {
    if (index != 0 && closureKey(units[index - 1]) == closureKey(units[index]))
      return invalid("duplicate programming unit resource closure");
    units[index].id = static_cast<ProgrammingUnitId>(index);
  }
  return CanonicalizedConfigurationABI{std::move(*system), std::move(units)};
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

std::string
serializeConfigurationABI(const ArtifactRootReference &fabricReference,
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
                json.attributeObject("semantic_encoding", [&] {
                  writeSemanticEncoding(json, field.semanticEncoding);
                });
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
                json.attribute("inactive_value", formatArtifactLocalPayloadHex(
                                                     field.inactiveValue));
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
  for (const ConfigurationFieldEncoding &field : unit.fields)
    if (llvm::ArrayRef<std::uint8_t>(referenceKey(field.slot)).equals(key))
      return &field;
  return nullptr;
}

llvm::Expected<ByteVector>
encodeField(const ConfigurationFieldEncoding &field,
            llvm::ArrayRef<std::uint8_t> semanticValue) {
  if (const auto *direct =
          std::get_if<DirectBitsEncoding>(&field.semanticEncoding)) {
    if (llvm::Error error = validateBitVector(
            semanticValue, direct->encodedBitCount, "DirectBits value"))
      return std::move(error);
    return ByteVector(semanticValue.begin(), semanticValue.end());
  }
  const auto &codebook =
      std::get<FiniteCodebookEncoding>(field.semanticEncoding);
  const auto entry = llvm::find_if(
      codebook.entries, [&](const FiniteCodebookEntry &candidate) {
        return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(semanticValue);
      });
  if (entry == codebook.entries.end())
    return invalid("finite codebook has no entry for semantic value");
  return entry->physicalCode;
}

} // namespace

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

  std::map<ByteVector, llvm::ArrayRef<std::uint8_t>> selected;
  for (const SemanticConfigurationValue &value : values) {
    ByteVector key = referenceKey(value.slot);
    if (!findFieldInUnit(*unit, key))
      return invalid(
          "semantic value names a field outside the programming unit");
    if (!selected.emplace(std::move(key), value.value).second)
      return invalid("semantic value names a configuration field twice");
  }

  for (const ConfigurationFieldEncoding &field : unit->fields) {
    ByteVector key = referenceKey(field.slot);
    auto found = selected.find(key);
    llvm::ArrayRef<std::uint8_t> semantic =
        found == selected.end()
            ? llvm::ArrayRef<std::uint8_t>(field.inactiveValue)
            : found->second;
    if (llvm::Error error = validateSemanticValue(
            system_, fabric::configurationField(field.slot), semantic))
      return std::move(error);
    auto encoded = encodeField(field, semantic);
    if (!encoded)
      return encoded.takeError();
    for (const DestinationSlice &slice : field.destinationSlices)
      for (std::uint64_t index = 0; index < slice.bitCount; ++index)
        setBit(payload, slice.destinationBitOffset + index,
               bit(*encoded, slice.sourceBitOffset + index));
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
    const std::uint64_t fieldBitCount = field.encodedBitCount();
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
    if (std::holds_alternative<DirectBitsEncoding>(field.semanticEncoding)) {
      semantic = std::move(physical);
    } else {
      const auto &codebook =
          std::get<FiniteCodebookEncoding>(field.semanticEncoding);
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
  const std::string json =
      serializeConfigurationABI(fabricReference, units->units);
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));

  auto reparsedDraft = parseConfigurationABI(bytes.bytes());
  if (!reparsedDraft)
    return reparsedDraft.takeError();
  ArtifactRootReference reparsedFabric = reparsedDraft->fabric;
  auto reparsedUnits = canonicalizeDraft(std::move(*reparsedDraft), store);
  if (!reparsedUnits)
    return reparsedUnits.takeError();
  if (serializeConfigurationABI(reparsedFabric, reparsedUnits->units) != json)
    return invalid("canonical JSON does not independently round-trip");

  auto identity = store.put(configurationAbiSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importConfigurationABI({configurationAbiSchema.identity.str(),
                                 configurationAbiSchema.version, *identity},
                                store);
}

llvm::Expected<FinalizedConfigurationABI>
importConfigurationABI(const ArtifactRootReference &reference,
                       const ArtifactStore &store) {
  if (reference.schemaIdentity != configurationAbiSchema.identity ||
      reference.schemaVersion != configurationAbiSchema.version)
    return invalid("reference is not loom.configuration_abi 3.0");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto draft = parseConfigurationABI(bytes->bytes());
  if (!draft)
    return draft.takeError();
  ArtifactRootReference fabricReference = draft->fabric;
  auto units = canonicalizeDraft(std::move(*draft), store);
  if (!units)
    return units.takeError();
  const std::string rewritten =
      serializeConfigurationABI(fabricReference, units->units);
  llvm::StringRef stored(reinterpret_cast<const char *>(bytes->bytes().data()),
                         bytes->bytes().size());
  if (stored != rewritten)
    return invalid("stored ConfigurationABI payload is not canonical");
  ConfigurationABI abi(fabricReference, std::move(units->units),
                       std::move(units->system));
  return FinalizedConfigurationABI(reference, std::move(*bytes),
                                   std::move(abi));
}

} // namespace loom::hardware

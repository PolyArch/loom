#include "Deployment/HardwareConfigurationImage.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/ConfiguredHardwareProjection.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::deployment {
namespace detail {

inline constexpr std::uint64_t configurationImageProjectionAlgorithmVersion =
    1;

struct ConfigurationImageProjectionSessionKey final {
  ArtifactRootReference configurationAbi;
  ConfigurationImageSourceKind sourceKind;
  ArtifactRootReference sourceMapping;
  std::optional<::loom::fabric::SpatialCoreOccurrenceRef> spatialOccurrence;
  std::uint64_t algorithmVersion =
      configurationImageProjectionAlgorithmVersion;

  friend bool operator==(const ConfigurationImageProjectionSessionKey &lhs,
                         const ConfigurationImageProjectionSessionKey &rhs) {
    return lhs.configurationAbi == rhs.configurationAbi &&
           lhs.sourceKind == rhs.sourceKind &&
           lhs.sourceMapping == rhs.sourceMapping &&
           lhs.spatialOccurrence == rhs.spatialOccurrence &&
           lhs.algorithmVersion == rhs.algorithmVersion;
  }
};

struct ConfigurationImageProjectionSessionEntry final {
  ConfigurationImageProjectionSessionKey key;
  std::shared_ptr<
      const mapping::PhysicalConfiguredHardwareProjectionView>
      projection;
  std::uint64_t retainedBytes = 0;
};

class ConfigurationImageProjectionSessionState final {
public:
  ConfigurationImageProjectionSessionState(const ArtifactStore &store,
                                            std::size_t entryLimit)
      : store_(&store), entryLimit_(entryLimit) {}

  bool owns(const ArtifactStore &store) const { return store_ == &store; }

  std::shared_ptr<const mapping::PhysicalConfiguredHardwareProjectionView>
  find(const ConfigurationImageProjectionSessionKey &key) {
    ++statistics_.requests;
    ++statistics_.deterministicWork;
    const auto found = llvm::find_if(
        entries_, [&](const auto &entry) { return entry.key == key; });
    if (found == entries_.end()) {
      ++statistics_.cacheMisses;
      return {};
    }
    ++statistics_.cacheHits;
    return found->projection;
  }

  std::shared_ptr<const mapping::PhysicalConfiguredHardwareProjectionView>
  insert(ConfigurationImageProjectionSessionKey key,
         std::shared_ptr<
             const mapping::PhysicalConfiguredHardwareProjectionView>
             projection,
         std::uint64_t retainedBytes,
         std::uint64_t constructionNanoseconds) {
    ++statistics_.uniqueConstructions;
    statistics_.constructionNanoseconds += constructionNanoseconds;
    statistics_.deterministicWork += projection->fields().size();
    if (entries_.size() >= entryLimit_) {
      ++statistics_.uncachedConstructions;
      return projection;
    }
    entries_.push_back({std::move(key), projection, retainedBytes});
    statistics_.retainedBytes += retainedBytes;
    statistics_.entryCount = entries_.size();
    return projection;
  }

  const FinalizedHardwareConfigurationImage *
  findImage(const ArtifactRootReference &reference) {
    ++statistics_.imageRequests;
    const auto found =
        llvm::find_if(imageEntries_, [&](const auto &entry) {
          return entry.first == reference;
        });
    if (found == imageEntries_.end()) {
      ++statistics_.imageCacheMisses;
      return nullptr;
    }
    ++statistics_.imageCacheHits;
    return &found->second;
  }

  void insertImage(const ArtifactRootReference &reference,
                   const FinalizedHardwareConfigurationImage &image) {
    if (imageEntries_.size() >= entryLimit_)
      return;
    imageEntries_.emplace_back(reference, image);
    statistics_.imageEntryCount = imageEntries_.size();
  }

  ConfigurationImageProjectionSessionStatistics statistics() const {
    return statistics_;
  }

private:
  const ArtifactStore *store_ = nullptr;
  std::size_t entryLimit_ = 0;
  std::vector<ConfigurationImageProjectionSessionEntry> entries_;
  std::vector<std::pair<ArtifactRootReference,
                        FinalizedHardwareConfigurationImage>>
      imageEntries_;
  ConfigurationImageProjectionSessionStatistics statistics_;
};

} // namespace detail
namespace {

using ByteVector = std::vector<std::uint8_t>;
using MonotonicClock = std::chrono::steady_clock;

thread_local detail::ConfigurationImageProjectionSessionState
    *currentProjectionSession = nullptr;

struct ParsedImage final {
  HardwareConfigurationImageDraft draft;
  std::uint64_t payloadBitCount = 0;
  ByteVector payload;
};

struct ParsedHeader final {
  HardwareConfigurationImageDraft draft;
  std::uint64_t payloadBitCount = 0;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_configuration_image_invalid: " +
                                     message);
}

llvm::Expected<std::uint32_t> readU32Be(llvm::ArrayRef<std::uint8_t> bytes,
                                        std::size_t &offset) {
  if (bytes.size() - offset < 4)
    return invalid("truncated canonical header size");
  const std::uint32_t value =
      (static_cast<std::uint32_t>(bytes[offset]) << 24) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
      static_cast<std::uint32_t>(bytes[offset + 3]);
  offset += 4;
  return value;
}

llvm::Expected<std::uint64_t> readU64Be(llvm::ArrayRef<std::uint8_t> bytes,
                                        std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated payload size");
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset + index];
  offset += 8;
  return value;
}

void appendU32Be(ByteVector &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(ByteVector &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Error
rejectUnknownFields(const llvm::json::Object &object, llvm::StringRef context,
                    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object) {
    if (std::find(allowed.begin(), allowed.end(),
                  llvm::StringRef(entry.first)) == allowed.end())
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(entry.first) + "'");
  }
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
  if (!schema)
    return schema.takeError();
  auto version = requireString(object, "schema_version", context);
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto artifact = requireString(object, "artifact", context);
  if (!artifact)
    return artifact.takeError();
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

llvm::StringRef sourceKindName(ConfigurationImageSourceKind kind) {
  switch (kind) {
  case ConfigurationImageSourceKind::SpatialMapping:
    return "spatial_mapping";
  case ConfigurationImageSourceKind::SystemMapping:
    return "system_mapping";
  }
  llvm_unreachable("unknown configuration image source kind");
}

llvm::Expected<ConfigurationImageSourceKind>
parseSourceKind(llvm::StringRef spelling) {
  if (spelling == "spatial_mapping")
    return ConfigurationImageSourceKind::SpatialMapping;
  if (spelling == "system_mapping")
    return ConfigurationImageSourceKind::SystemMapping;
  return invalid("source_mapping has unknown kind '" + spelling + "'");
}

std::string serializeHeader(const HardwareConfigurationImageDraft &draft,
                            std::uint64_t payloadBitCount) {
  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", hardwareConfigurationImageSchema.identity);
    json.attribute(
        "schema_version",
        formatSchemaVersion(hardwareConfigurationImageSchema.version));
    json.attributeBegin("configuration_abi_ref");
    json.object([&] { writeRootReference(json, draft.configurationAbi); });
    json.attributeEnd();
    json.attribute("programming_unit_id", draft.programmingUnitId);
    json.attributeBegin("source_mapping");
    json.object([&] {
      json.attribute("kind", sourceKindName(draft.sourceMapping.kind));
      json.attributeBegin("reference");
      json.object(
          [&] { writeRootReference(json, draft.sourceMapping.mapping); });
      json.attributeEnd();
    });
    json.attributeEnd();
    json.attribute("payload_bit_count", payloadBitCount);
  });
  return output.str().str();
}

llvm::Expected<ParsedHeader> parseHeader(llvm::StringRef header) {
  auto parsed = llvm::json::parse(header);
  if (!parsed)
    return invalid("canonical header is not valid JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("canonical header must be a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "canonical header",
          {"schema", "schema_version", "configuration_abi_ref",
           "programming_unit_id", "source_mapping", "payload_bit_count"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "canonical header");
  if (!schema)
    return schema.takeError();
  auto version = requireString(*root, "schema_version", "canonical header");
  if (!version)
    return version.takeError();
  if (*schema != hardwareConfigurationImageSchema.identity ||
      *version != formatSchemaVersion(hardwareConfigurationImageSchema.version))
    return invalid("canonical header has the wrong schema descriptor");

  auto abiObject =
      requireObject(*root, "configuration_abi_ref", "canonical header");
  if (!abiObject)
    return abiObject.takeError();
  auto abi = parseRootReference(**abiObject, "configuration_abi_ref");
  if (!abi)
    return abi.takeError();
  auto unit = requireUnsigned(*root, "programming_unit_id", "canonical header");
  if (!unit)
    return unit.takeError();
  auto sourceObject =
      requireObject(*root, "source_mapping", "canonical header");
  if (!sourceObject)
    return sourceObject.takeError();
  if (llvm::Error error = rejectUnknownFields(**sourceObject, "source_mapping",
                                              {"kind", "reference"}))
    return std::move(error);
  auto kind = requireString(**sourceObject, "kind", "source_mapping");
  if (!kind)
    return kind.takeError();
  auto parsedKind = parseSourceKind(*kind);
  if (!parsedKind)
    return parsedKind.takeError();
  auto mappingObject =
      requireObject(**sourceObject, "reference", "source_mapping");
  if (!mappingObject)
    return mappingObject.takeError();
  auto mapping =
      parseRootReference(**mappingObject, "source_mapping.reference");
  if (!mapping)
    return mapping.takeError();
  if (mapping->schemaIdentity != mapping::mappingArtifactSchema.identity ||
      mapping->schemaVersion != mapping::mappingArtifactSchema.version)
    return invalid("source_mapping does not reference the Mapping schema");
  auto payloadBitCount =
      requireUnsigned(*root, "payload_bit_count", "canonical header");
  if (!payloadBitCount)
    return payloadBitCount.takeError();
  return ParsedHeader{
      HardwareConfigurationImageDraft{
          std::move(*abi), *unit,
          ConfigurationImageSourceRef{*parsedKind, std::move(*mapping)}},
      *payloadBitCount};
}

llvm::Expected<ParsedImage> parseImage(llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  auto headerSize = readU32Be(bytes, offset);
  if (!headerSize)
    return headerSize.takeError();
  if (*headerSize > bytes.size() - offset)
    return invalid("canonical header is truncated");
  const llvm::StringRef header(
      reinterpret_cast<const char *>(bytes.data() + offset), *headerSize);
  offset += *headerSize;
  auto parsedHeader = parseHeader(header);
  if (!parsedHeader)
    return parsedHeader.takeError();
  auto payloadSize = readU64Be(bytes, offset);
  if (!payloadSize)
    return payloadSize.takeError();
  if (*payloadSize != bytes.size() - offset)
    return invalid("payload size does not match the exact trailing bytes");

  ByteVector payload(bytes.begin() + offset, bytes.end());
  return ParsedImage{std::move(parsedHeader->draft),
                     parsedHeader->payloadBitCount, std::move(payload)};
}

CanonicalSemanticBytes frameImage(const HardwareConfigurationImageDraft &draft,
                                  std::uint64_t payloadBitCount,
                                  llvm::ArrayRef<std::uint8_t> payload) {
  const std::string header = serializeHeader(draft, payloadBitCount);
  ByteVector bytes;
  bytes.reserve(4 + header.size() + 8 + payload.size());
  appendU32Be(bytes, static_cast<std::uint32_t>(header.size()));
  bytes.insert(bytes.end(), header.begin(), header.end());
  appendU64Be(bytes, payload.size());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<::loom::fabric::SpatialCoreOccurrenceRef>
spatialOccurrence(const hardware::ProgrammingUnit &unit) {
  std::optional<::loom::fabric::SpatialCoreOccurrenceRef> result;
  for (const hardware::ConfigurationFieldEncoding &field : unit.fields) {
    if (field.slot.kind() !=
        ::loom::fabric::FabricPhysicalConfigurationSlotKind::
            SpatialCoreInternalSlot)
      continue;
    const auto &internal =
        std::get<::loom::fabric::SpatialCoreInternalConfigurationSlotRef>(
            field.slot.payload());
    if (result && *result != internal.spatialCore)
      return invalid("one programming unit spans multiple SpatialCore "
                     "occurrences");
    result = internal.spatialCore;
  }
  if (!result)
    return invalid("SpatialMapping source requires a SpatialCore-local "
                   "programming unit");
  return *result;
}

llvm::Expected<mapping::PhysicalConfiguredHardwareProjectionView>
configurationProjection(const HardwareConfigurationImageDraft &draft,
                        const hardware::ConfigurationABI &abi,
                        const hardware::ProgrammingUnit &unit,
                        const ArtifactStore &store) {
  switch (draft.sourceMapping.kind) {
  case ConfigurationImageSourceKind::SpatialMapping: {
    auto mapping =
        mapping::importSpatialMapping(draft.sourceMapping.mapping, store);
    if (!mapping)
      return mapping.takeError();
    auto occurrence = spatialOccurrence(unit);
    if (!occurrence)
      return occurrence.takeError();
    return mapping::qualifyConfiguredHardwareProjection(
        *mapping, abi.fabricSystem(), *occurrence);
  }
  case ConfigurationImageSourceKind::SystemMapping: {
    auto mapping =
        mapping::importSystemMapping(draft.sourceMapping.mapping, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().fabricIdentity() !=
        abi.fabricSystem().artifact().identity())
      return invalid("SystemMapping and ConfigurationABI bind different "
                     "Fabric systems");
    return mapping::deriveConfiguredHardwareProjection(*mapping, store);
  }
  }
  llvm_unreachable("unknown configuration image source kind");
}

llvm::Expected<detail::ConfigurationImageProjectionSessionKey>
configurationProjectionKey(const HardwareConfigurationImageDraft &draft,
                           const hardware::FinalizedConfigurationABI &abi,
                           const hardware::ProgrammingUnit &unit) {
  std::optional<::loom::fabric::SpatialCoreOccurrenceRef> occurrence;
  if (draft.sourceMapping.kind ==
      ConfigurationImageSourceKind::SpatialMapping) {
    auto resolved = spatialOccurrence(unit);
    if (!resolved)
      return resolved.takeError();
    occurrence = *resolved;
  }
  return detail::ConfigurationImageProjectionSessionKey{
      abi.reference(), draft.sourceMapping.kind, draft.sourceMapping.mapping,
      occurrence};
}

std::uint64_t retainedProjectionBytes(
    const mapping::PhysicalConfiguredHardwareProjectionView &projection) {
  std::uint64_t bytes = sizeof(projection);
  bytes += projection.fields().size() *
           sizeof(mapping::PhysicalConfiguredHardwareFieldValueView);
  for (const auto &field : projection.fields())
    bytes += field.value.bytes().size();
  return bytes;
}

llvm::Expected<
    std::shared_ptr<const mapping::PhysicalConfiguredHardwareProjectionView>>
resolveConfigurationProjection(
    const HardwareConfigurationImageDraft &draft,
    const hardware::FinalizedConfigurationABI &abi,
    const hardware::ProgrammingUnit &unit, const ArtifactStore &store) {
  auto key = configurationProjectionKey(draft, abi, unit);
  if (!key)
    return key.takeError();
  if (currentProjectionSession && !currentProjectionSession->owns(store))
    return invalid("configuration projection session crosses its ArtifactStore "
                   "verification domain");
  if (currentProjectionSession)
    if (auto cached = currentProjectionSession->find(*key))
      return cached;

  const auto begin = MonotonicClock::now();
  auto projected = configurationProjection(draft, abi.abi(), unit, store);
  if (!projected)
    return projected.takeError();
  auto result = std::make_shared<
      const mapping::PhysicalConfiguredHardwareProjectionView>(
      std::move(*projected));
  if (!currentProjectionSession)
    return result;
  const std::uint64_t constructionNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          MonotonicClock::now() - begin)
          .count();
  return currentProjectionSession->insert(
      std::move(*key), result, retainedProjectionBytes(*result),
      constructionNanoseconds);
}

llvm::Expected<ByteVector>
derivePayload(const HardwareConfigurationImageDraft &draft,
              const hardware::FinalizedConfigurationABI &finalizedAbi,
              const mapping::PhysicalConfiguredHardwareProjectionView
                  &projection) {
  const hardware::ProgrammingUnit *unit =
      finalizedAbi.abi().findProgrammingUnit(draft.programmingUnitId);
  if (!unit)
    return invalid("configuration_abi_ref has no programming_unit_id");

  std::vector<hardware::SemanticConfigurationValue> values;
  for (const mapping::PhysicalConfiguredHardwareFieldValueView &field :
       projection.fields()) {
    if (finalizedAbi.abi().findField(unit->id, field.slot)) {
      values.push_back({field.slot, ByteVector(field.value.bytes().begin(),
                                               field.value.bytes().end())});
      continue;
    }
    if (!finalizedAbi.abi().findField(field.slot))
      return invalid("Mapping projection names a field outside the exact ABI");
  }
  return finalizedAbi.abi().encode(draft.programmingUnitId, values);
}

llvm::Expected<hardware::FinalizedConfigurationABI>
importAbi(const HardwareConfigurationImageDraft &draft,
          const ArtifactStore &store) {
  if (draft.configurationAbi.schemaIdentity !=
          hardware::configurationAbiSchema.identity ||
      draft.configurationAbi.schemaVersion !=
          hardware::configurationAbiSchema.version)
    return invalid("configuration_abi_ref has the wrong schema descriptor");
  return hardware::importConfigurationABI(draft.configurationAbi, store);
}

llvm::StringRef spelling(
    ConfigurationImageProjectionVerificationDomain domain) {
  switch (domain) {
  case ConfigurationImageProjectionVerificationDomain::SourceInvocation:
    return "source_invocation";
  case ConfigurationImageProjectionVerificationDomain::IndependentReplay:
    return "independent_replay";
  }
  llvm_unreachable("unknown configuration image projection domain");
}

} // namespace

ConfigurationImageProjectionSession::ConfigurationImageProjectionSession(
    const ArtifactStore &store, std::size_t entryLimit)
    : state_(std::make_unique<
             detail::ConfigurationImageProjectionSessionState>(store,
                                                               entryLimit)),
      previous_(currentProjectionSession) {
  currentProjectionSession = state_.get();
}

ConfigurationImageProjectionSession::~ConfigurationImageProjectionSession() {
  currentProjectionSession = previous_;
}

ConfigurationImageProjectionSessionStatistics
ConfigurationImageProjectionSession::statistics() const {
  return state_->statistics();
}

void emitConfigurationImageProjectionSessionStatistics(
    ConfigurationImageProjectionVerificationDomain domain,
    const ConfigurationImageProjectionSessionStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::ConfigurationImageProjectionSession, [&] {
        llvm::json::Object payload;
        payload["verification_domain"] = spelling(domain);
        payload["requests"] = statistics.requests;
        payload["cache_hits"] = statistics.cacheHits;
        payload["cache_misses"] = statistics.cacheMisses;
        payload["unique_constructions"] = statistics.uniqueConstructions;
        payload["uncached_constructions"] =
            statistics.uncachedConstructions;
        payload["construction_time_ns"] =
            statistics.constructionNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["retained_bytes"] = statistics.retainedBytes;
        payload["entry_count"] = statistics.entryCount;
        payload["image_requests"] = statistics.imageRequests;
        payload["image_cache_hits"] = statistics.imageCacheHits;
        payload["image_cache_misses"] = statistics.imageCacheMisses;
        payload["image_entry_count"] = statistics.imageEntryCount;
        return llvm::json::Value(std::move(payload));
      });
}

llvm::Expected<FinalizedHardwareConfigurationImage>
finalizeHardwareConfigurationImage(HardwareConfigurationImageDraft draft,
                                   const ArtifactStore &store) {
  auto abi = importAbi(draft, store);
  if (!abi)
    return abi.takeError();
  const hardware::ProgrammingUnit *unit =
      abi->abi().findProgrammingUnit(draft.programmingUnitId);
  if (!unit)
    return invalid("configuration_abi_ref has no programming_unit_id");
  auto projection = resolveConfigurationProjection(draft, *abi, *unit, store);
  if (!projection)
    return projection.takeError();
  auto payload = derivePayload(draft, *abi, **projection);
  if (!payload)
    return payload.takeError();
  CanonicalSemanticBytes bytes =
      frameImage(draft, unit->payloadBitCount, *payload);
  auto identity = store.put(hardwareConfigurationImageSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importHardwareConfigurationImage(
      {hardwareConfigurationImageSchema.identity.str(),
       hardwareConfigurationImageSchema.version, *identity},
      store);
}

llvm::Expected<FinalizedHardwareConfigurationImage>
importHardwareConfigurationImage(const ArtifactRootReference &reference,
                                 const ArtifactStore &store) {
  if (reference.schemaIdentity != hardwareConfigurationImageSchema.identity ||
      reference.schemaVersion != hardwareConfigurationImageSchema.version)
    return invalid("root reference has the wrong schema descriptor");
  detail::ConfigurationImageProjectionSessionState *session =
      currentProjectionSession && currentProjectionSession->owns(store)
          ? currentProjectionSession
          : nullptr;
  if (session)
    if (const FinalizedHardwareConfigurationImage *cached =
            session->findImage(reference))
      return *cached;
  auto bytes = store.get(hardwareConfigurationImageSchema, reference.artifact);
  if (!bytes)
    return bytes.takeError();
  auto parsed = parseImage(bytes->bytes());
  if (!parsed)
    return parsed.takeError();
  auto abi = importAbi(parsed->draft, store);
  if (!abi)
    return abi.takeError();
  const hardware::ProgrammingUnit *unit =
      abi->abi().findProgrammingUnit(parsed->draft.programmingUnitId);
  if (!unit)
    return invalid("configuration_abi_ref has no programming_unit_id");
  if (parsed->payloadBitCount != unit->payloadBitCount)
    return invalid("payload_bit_count disagrees with the programming unit");
  auto projection =
      resolveConfigurationProjection(parsed->draft, *abi, *unit, store);
  if (!projection)
    return projection.takeError();
  auto expected = derivePayload(parsed->draft, *abi, **projection);
  if (!expected)
    return expected.takeError();
  if (*expected != parsed->payload)
    return invalid("payload does not encode the exact Mapping projection");
  const CanonicalSemanticBytes canonical =
      frameImage(parsed->draft, parsed->payloadBitCount, parsed->payload);
  if (!canonical.bytes().equals(bytes->bytes()))
    return invalid("stored image payload is not canonical");
  FinalizedHardwareConfigurationImage finalized(
      reference, std::move(*bytes),
      HardwareConfigurationImage(std::move(parsed->draft.configurationAbi),
                                 parsed->draft.programmingUnitId,
                                 std::move(parsed->draft.sourceMapping),
                                 parsed->payloadBitCount,
                                 std::move(parsed->payload)));
  if (session)
    session->insertImage(reference, finalized);
  return finalized;
}

} // namespace loom::deployment

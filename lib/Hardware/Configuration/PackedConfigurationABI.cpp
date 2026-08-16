#include "Hardware/Configuration/PackedConfigurationABI.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware {
namespace {

inline constexpr std::uint64_t packedRelationDerivationVersion = 1;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "packed_configuration_abi_invalid: " +
                                     message);
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

llvm::Expected<fabric::FabricPhysicalOccurrenceOwnerRef>
qualifyOwner(fabric::SpatialCoreOccurrenceRef spatialCore,
             const fabric::FabricModulePhysicalOwnerRef &owner) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(owner);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalOccurrenceOwnerRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyField(fabric::SpatialCoreOccurrenceRef spatialCore,
             const fabric::FabricSemanticConfigFieldRef &field) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(field);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

llvm::Expected<fabric::FabricArtifactView>
importedModule(const fabric::FabricSystemRootView &system,
               fabric::SpatialCoreOccurrenceRef spatialCore) {
  const auto target = system.spatialCoreTarget(spatialCore.core);
  if (!target)
    return invalid("SpatialCore has no imported Module target");
  const auto modules = system.artifact().importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalid("SpatialCore imported Module dependency is out of range");
  const fabric::FabricArtifactView &module =
      modules[static_cast<std::size_t>(target->dependencyOrdinal)];
  if (module.moduleRootTemplate() != target->target)
    return invalid("SpatialCore imported Module target does not match");
  return module;
}

std::vector<std::uint8_t> bitVector(std::uint64_t value,
                                    std::uint64_t bitCount) {
  std::vector<std::uint8_t> result((bitCount + 7) / 8, 0);
  for (std::uint64_t bit = 0; bit < bitCount; ++bit)
    if ((value & (std::uint64_t(1) << bit)) != 0)
      result[static_cast<std::size_t>(bit / 8)] |=
          static_cast<std::uint8_t>(1U << (bit % 8));
  return result;
}

std::uint64_t codebookBitCount(std::size_t domainSize) {
  std::uint64_t bits = 1;
  while ((std::uint64_t(1) << bits) <= domainSize)
    ++bits;
  return bits;
}

llvm::Expected<ConfigurationEncodingRelationDraft>
defaultFieldEncoding(const fabric::FabricArtifactView &module,
                     const fabric::FabricSemanticConfigFieldRef &localField,
                     mlir::MLIRContext &context) {
  auto relation = module.semanticFieldRelation(localField, context);
  if (!relation)
    return relation.takeError();
  if (relation->kind() == fabric::FabricSemanticFieldRelationKind::Direct) {
    const auto width = relation->directEncodedBitCount();
    if (!width || *width == 0)
      return invalid("direct configuration field has no encoded width");
    const CanonicalSemanticBytes *canonicalInactive =
        relation->canonicalInactiveValue();
    if (!canonicalInactive)
      return invalid("direct configuration field has no canonical inactive "
                     "value");
    std::vector<std::uint8_t> inactive(canonicalInactive->bytes().begin(),
                                       canonicalInactive->bytes().end());
    if (llvm::Error error = relation->validateSemanticValue(inactive))
      return std::move(error);
    return ConfigurationEncodingRelationDraft{DirectBitsEncoding{*width},
                                              std::move(inactive)};
  }
  if (relation->kind() != fabric::FabricSemanticFieldRelationKind::Finite)
    return invalid("configuration field has no semantic relation");

  std::vector<std::vector<std::uint8_t>> values;
  values.reserve(relation->finiteDomain().size());
  for (const CanonicalSemanticBytes &value : relation->finiteDomain())
    values.emplace_back(value.bytes().begin(), value.bytes().end());
  if (values.empty())
    return invalid("finite configuration domain is empty");
  const std::set<std::vector<std::uint8_t>> unique(values.begin(),
                                                   values.end());
  if (unique.size() != values.size())
    return invalid("configuration field domain contains duplicate carriers");

  const std::uint64_t bitCount = codebookBitCount(values.size());
  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(values.size());
  for (const auto &[ordinal, value] : llvm::enumerate(values))
    entries.push_back(
        {value, bitVector(static_cast<std::uint64_t>(ordinal), bitCount)});
  return ConfigurationEncodingRelationDraft{
      FiniteCodebookEncoding{bitCount, std::move(entries)}, values.front()};
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 64; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (shift - 8)));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<std::uint8_t>
relationKey(const ConfigurationEncodingRelationDraft &relation) {
  std::vector<std::uint8_t> key;
  if (const auto *direct =
          std::get_if<DirectBitsEncoding>(&relation.semanticEncoding)) {
    key.push_back(0);
    appendU64(key, direct->encodedBitCount);
  } else {
    key.push_back(1);
    const auto &codebook =
        std::get<FiniteCodebookEncoding>(relation.semanticEncoding);
    appendU64(key, codebook.encodedBitCount);
    std::vector<FiniteCodebookEntry> entries = codebook.entries;
    llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.semanticValue, lhs.physicalCode) <
             std::tie(rhs.semanticValue, rhs.physicalCode);
    });
    appendU64(key, entries.size());
    for (const FiniteCodebookEntry &entry : entries) {
      appendFramed(key, entry.semanticValue);
      appendFramed(key, entry.physicalCode);
    }
  }
  appendFramed(key, relation.inactiveValue);
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
relationSourceKey(const fabric::FabricArtifactView &module,
                  const fabric::FabricSemanticConfigFieldRef &field) {
  std::vector<std::uint8_t> key(module.identity().bytes().begin(),
                                module.identity().bytes().end());
  appendFramed(key, llvm::ArrayRef<std::uint8_t>(
                        reinterpret_cast<const std::uint8_t *>(
                            configurationAbiSchema.identity.data()),
                        configurationAbiSchema.identity.size()));
  appendU64(key, configurationAbiSchema.version.major);
  appendU64(key, configurationAbiSchema.version.minor);
  appendU64(key, packedRelationDerivationVersion);

  auto source = fabric::semanticFieldRelationSourceIdentity(module, field);
  if (!source)
    return source.takeError();
  appendFramed(key, source->bytes());
  return key;
}

using OverrideKey = std::vector<std::uint8_t>;

OverrideKey
overrideKey(const fabric::FabricPhysicalConfigurationFieldRef &field) {
  return fabric::canonicalFabricBytes(field);
}

} // namespace

llvm::Expected<ConfigurationABIDraft> derivePackedConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &systemRoot, mlir::MLIRContext &context,
    llvm::ArrayRef<PackedConfigurationFieldEncodingOverride> overrides,
    PackedConfigurationABIDerivationStatistics *statistics) {
  const auto started = std::chrono::steady_clock::now();
  auto system = fabric::requireSystemRoot(systemRoot.view());
  if (!system)
    return system.takeError();

  std::map<OverrideKey, const PackedConfigurationFieldEncodingOverride *>
      overrideByField;
  for (const PackedConfigurationFieldEncodingOverride &override : overrides)
    if (!overrideByField.emplace(overrideKey(override.field), &override).second)
      return invalid("configuration field override is duplicated");
  std::set<OverrideKey> consumedOverrides;

  std::vector<ConfigurationEncodingRelationDraft> encodingRelations;
  std::map<std::vector<std::uint8_t>, ConfigurationEncodingRelationId>
      relationByKey;
  std::map<std::vector<std::uint8_t>, ConfigurationEncodingRelationId>
      relationBySource;
  std::uint64_t sourceCacheHits = 0;
  std::uint64_t sourceCacheMisses = 0;
  std::uint64_t relationCacheHits = 0;
  std::uint64_t relationCacheMisses = 0;
  std::uint64_t deterministicWork = overrides.size();
  std::uint64_t configurationFieldCount = 0;
  const auto internRelation = [&](ConfigurationEncodingRelationDraft relation)
      -> ConfigurationEncodingRelationId {
    std::vector<std::uint8_t> key = relationKey(relation);
    auto found = relationByKey.find(key);
    if (found != relationByKey.end()) {
      ++relationCacheHits;
      return found->second;
    }
    ++relationCacheMisses;
    ++deterministicWork;
    const auto id =
        static_cast<ConfigurationEncodingRelationId>(encodingRelations.size());
    encodingRelations.push_back(std::move(relation));
    relationByKey.emplace(std::move(key), id);
    return id;
  };

  std::vector<ProgrammingUnitDraft> units;
  for (fabric::AccCoreOccurrenceRef core :
       system->artifact().accCoreOccurrences()) {
    const fabric::SpatialCoreOccurrenceRef spatialCore{core};
    auto module = importedModule(*system, spatialCore);
    if (!module)
      return module.takeError();

    std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> closure;
    std::vector<ConfigurationFieldEncoding> fields;
    for (const fabric::FabricModuleDomainMemberRef &member :
         module->moduleDomainMembers()) {
      ++deterministicWork;
      if (member.kind() != fabric::FabricModuleDomainMemberKind::Internal)
        continue;
      const auto &localOwner =
          std::get<fabric::FabricModulePhysicalOwnerRef>(member.payload);
      const fabric::FabricInventoryOwnerRef owner = inventoryOwner(localOwner);
      const std::uint64_t fieldCount = module->inventorySize(
          owner, fabric::FabricInventoryKind::SemanticConfigField);
      if (fieldCount == 0)
        continue;
      auto physicalOwner = qualifyOwner(spatialCore, localOwner);
      if (!physicalOwner)
        return physicalOwner.takeError();
      closure.push_back(std::move(*physicalOwner));
      for (fabric::FabricOrdinal ordinal = 0; ordinal < fieldCount; ++ordinal) {
        const fabric::FabricSemanticConfigFieldRef localField{
            fabric::FabricConfigurationOwnerRef(owner), ordinal};
        auto physicalField = qualifyField(spatialCore, localField);
        if (!physicalField)
          return physicalField.takeError();
        const OverrideKey key = overrideKey(*physicalField);
        const auto override = overrideByField.find(key);
        ConfigurationEncodingRelationId relationId = 0;
        if (override != overrideByField.end()) {
          relationId = internRelation(ConfigurationEncodingRelationDraft{
              override->second->semanticEncoding,
              override->second->inactiveValue});
        } else {
          auto sourceKey = relationSourceKey(*module, localField);
          if (!sourceKey)
            return sourceKey.takeError();
          const auto known = relationBySource.find(*sourceKey);
          if (known != relationBySource.end()) {
            ++sourceCacheHits;
            relationId = known->second;
          } else {
            ++sourceCacheMisses;
            auto relation = defaultFieldEncoding(*module, localField, context);
            if (!relation)
              return relation.takeError();
            relationId = internRelation(std::move(*relation));
            relationBySource.emplace(std::move(*sourceKey), relationId);
          }
        }
        auto residencies = module->configurationResidencies(localField);
        if (!residencies)
          return residencies.takeError();
        for (const fabric::FabricConfigurationResidency &residency :
             *residencies) {
          ++configurationFieldCount;
          ++deterministicWork;
          auto slot =
              fabric::qualifyFabricConfigurationSlot(*physicalField, residency);
          if (!slot)
            return slot.takeError();
          fields.push_back(
              ConfigurationFieldEncoding{std::move(*slot), relationId, {}});
        }
        if (override != overrideByField.end())
          consumedOverrides.insert(key);
      }
    }
    if (fields.empty())
      continue;

    std::uint64_t payloadBitCount = 1;
    for (ConfigurationFieldEncoding &field : fields) {
      const std::uint64_t width =
          encodingRelations[field.encodingRelation].encodedBitCount();
      field.destinationSlices.push_back({0, payloadBitCount, width});
      payloadBitCount += width;
    }
    units.push_back(ProgrammingUnitDraft{std::move(closure), payloadBitCount,
                                         std::move(fields)});
  }

  if (consumedOverrides.size() != overrideByField.size())
    return invalid("configuration field override does not name a System field");
  if (statistics) {
    std::uint64_t retainedCacheBytes = 0;
    for (const auto &entry : relationByKey)
      retainedCacheBytes += sizeof(entry) + entry.first.size();
    for (const auto &entry : relationBySource)
      retainedCacheBytes += sizeof(entry) + entry.first.size();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    *statistics = PackedConfigurationABIDerivationStatistics{
        1,
        static_cast<std::uint64_t>(elapsed.count()),
        sourceCacheHits,
        sourceCacheMisses,
        relationCacheHits,
        relationCacheMisses,
        retainedCacheBytes,
        deterministicWork,
        units.size(),
        configurationFieldCount,
        encodingRelations.size()};
  }
  return ConfigurationABIDraft{systemRoot.reference(),
                               std::move(encodingRelations), std::move(units)};
}

} // namespace loom::hardware

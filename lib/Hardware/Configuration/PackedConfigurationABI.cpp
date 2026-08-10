#include "Hardware/Configuration/PackedConfigurationABI.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware {
namespace {

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

llvm::Expected<ConfigurationFieldEncoding>
defaultFieldEncoding(const fabric::FabricArtifactView &module,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricSemanticConfigFieldRef &localField,
                     fabric::FabricConfigurationResidency residency,
                     mlir::MLIRContext &context) {
  auto physical = qualifyField(spatialCore, localField);
  if (!physical)
    return physical.takeError();
  auto slot =
      fabric::qualifyFabricConfigurationSlot(*physical, std::move(residency));
  if (!slot)
    return slot.takeError();
  auto relation = module.semanticFieldRelation(localField, context);
  if (!relation)
    return relation.takeError();
  if (relation->kind() != fabric::FabricSemanticFieldRelationKind::Finite)
    return invalid("direct configuration field requires an explicit override");

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
  return ConfigurationFieldEncoding{
      std::move(*slot),
      FiniteCodebookEncoding{bitCount, std::move(entries)},
      {},
      values.front()};
}

using OverrideKey = std::vector<std::uint8_t>;

OverrideKey
overrideKey(const fabric::FabricPhysicalConfigurationFieldRef &field) {
  return fabric::canonicalFabricBytes(field);
}

} // namespace

llvm::Expected<ConfigurationABIDraft> derivePackedConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &systemRoot, mlir::MLIRContext &context,
    llvm::ArrayRef<PackedConfigurationFieldEncodingOverride> overrides) {
  auto system = fabric::requireSystemRoot(systemRoot.view());
  if (!system)
    return system.takeError();

  std::map<OverrideKey, const PackedConfigurationFieldEncodingOverride *>
      overrideByField;
  for (const PackedConfigurationFieldEncodingOverride &override : overrides)
    if (!overrideByField.emplace(overrideKey(override.field), &override).second)
      return invalid("configuration field override is duplicated");
  std::set<OverrideKey> consumedOverrides;

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
        auto residencies = module->configurationResidencies(localField);
        if (!residencies)
          return residencies.takeError();
        for (const fabric::FabricConfigurationResidency &residency :
             *residencies) {
          if (override != overrideByField.end()) {
            auto slot = fabric::qualifyFabricConfigurationSlot(
                override->second->field, residency);
            if (!slot)
              return slot.takeError();
            fields.push_back(
                ConfigurationFieldEncoding{std::move(*slot),
                                           override->second->semanticEncoding,
                                           {},
                                           override->second->inactiveValue});
            continue;
          }
          auto field = defaultFieldEncoding(*module, spatialCore, localField,
                                            residency, context);
          if (!field)
            return field.takeError();
          fields.push_back(std::move(*field));
        }
        if (override != overrideByField.end())
          consumedOverrides.insert(key);
      }
    }
    if (fields.empty())
      continue;

    std::uint64_t payloadBitCount = 1;
    for (ConfigurationFieldEncoding &field : fields) {
      const std::uint64_t width = field.encodedBitCount();
      field.destinationSlices.push_back({0, payloadBitCount, width});
      payloadBitCount += width;
    }
    units.push_back(ProgrammingUnitDraft{std::move(closure), payloadBitCount,
                                         std::move(fields)});
  }

  if (consumedOverrides.size() != overrideByField.size())
    return invalid("configuration field override does not name a System field");
  return ConfigurationABIDraft{systemRoot.reference(), std::move(units)};
}

} // namespace loom::hardware

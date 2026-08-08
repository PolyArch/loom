#include "ConfigurationABI2TestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::test {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

mlir::MLIRContext &relationContext() {
  static mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *context;
}

llvm::Expected<fabric::InstructionCoreMicroarchitecturalRealization>
makeInstructionCoreMicroarchitecture() {
  fabric::InstructionCoreCommonDeclaration common{
      1,
      {{fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1, 1, 1, 2, 1};
  return fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
      std::move(common), pipeline);
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

struct FieldDomain final {
  std::vector<std::vector<std::uint8_t>> values;
  std::vector<std::uint8_t> inactiveValue;
};

llvm::Expected<FieldDomain>
finiteFieldDomain(const fabric::FabricArtifactView &module,
                  const fabric::FabricSemanticConfigFieldRef &field) {
  const fabric::FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() == fabric::FabricInventoryOwnerKind::PeOccurrence) {
    auto schema = module.spatialPeConfigurationSchema(
        std::get<fabric::FabricPeOccurrenceRef>(owner.payload));
    if (!schema)
      return schema.takeError();
    auto domain = schema->finiteDomain(field);
    if (!domain)
      return domain.takeError();
    FieldDomain result;
    for (const fabric::FabricPeConfigurationValue &value : *domain) {
      auto encoded = schema->encode(field, value);
      if (!encoded)
        return encoded.takeError();
      result.values.emplace_back(encoded->bytes().begin(),
                                 encoded->bytes().end());
    }
    const auto descriptor =
        llvm::find_if(schema->fields(), [&](const auto &candidate) {
          return candidate.reference == field;
        });
    if (descriptor == schema->fields().end())
      return invalid("PE configuration field is absent from its schema");
    if (descriptor->kind ==
        fabric::FabricPeConfigurationFieldKind::Activation) {
      auto disabled = schema->encode(field, fabric::FabricPeDisabled{});
      if (!disabled)
        return disabled.takeError();
      result.inactiveValue.assign(disabled->bytes().begin(),
                                  disabled->bytes().end());
    } else {
      if (result.values.empty())
        return invalid("PE configuration field has an empty domain");
      result.inactiveValue = result.values.front();
    }
    return result;
  }

  if (owner.kind() != fabric::FabricInventoryOwnerKind::FuOccurrenceNode)
    return invalid("test support cannot derive this configuration field");
  const auto occurrence =
      std::get<fabric::FabricFuOccurrenceNodeRef>(owner.payload);
  const fabric::ResolvedFabricOpCapabilityView *capability =
      module.resolvedFabricOpCapability(occurrence);
  if (!capability)
    return invalid("operation configuration field has no capability");
  auto relation = capability->resolveSemanticFieldRelation(relationContext());
  if (!relation)
    return relation.takeError();
  if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
    return invalid("direct operation field requires an explicit override");

  FieldDomain result;
  for (const auto &point : relation->finiteBehaviorDomain()) {
    if (!point.semanticConfiguration)
      return invalid("finite operation behavior has no semantic carrier");
    result.values.emplace_back(point.semanticConfiguration->bytes().begin(),
                               point.semanticConfiguration->bytes().end());
  }
  if (result.values.empty())
    return invalid("finite operation configuration domain is empty");
  result.inactiveValue = result.values.front();
  return result;
}

llvm::Expected<ConfigurationFieldEncoding>
defaultFieldEncoding(const fabric::FabricArtifactView &module,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricSemanticConfigFieldRef &localField) {
  auto physical = qualifyField(spatialCore, localField);
  if (!physical)
    return physical.takeError();
  auto domain = finiteFieldDomain(module, localField);
  if (!domain)
    return domain.takeError();

  std::set<std::vector<std::uint8_t>> unique(domain->values.begin(),
                                             domain->values.end());
  if (unique.size() != domain->values.size())
    return invalid("configuration field domain contains duplicate carriers");
  if (!unique.count(domain->inactiveValue))
    return invalid("configuration field inactive value is outside its domain");
  const std::uint64_t bitCount = codebookBitCount(domain->values.size());
  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain->values.size());
  for (const auto &[ordinal, value] : llvm::enumerate(domain->values))
    entries.push_back(
        {value, bitVector(static_cast<std::uint64_t>(ordinal), bitCount)});
  return ConfigurationFieldEncoding{
      std::move(*physical),
      FiniteCodebookEncoding{bitCount, std::move(entries)},
      {},
      std::move(domain->inactiveValue)};
}

using OverrideKey = std::vector<std::uint8_t>;

OverrideKey
overrideKey(const fabric::FabricPhysicalConfigurationFieldRef &field) {
  return fabric::canonicalFabricBytes(field);
}

} // namespace

llvm::Expected<fabric::FinalizedFabricRoot>
makeSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                      const ArtifactStore &store,
                      std::uint64_t spatialCoreCount) {
  if (spatialCoreCount == 0)
    return invalid("test System requires at least one SpatialCore");
  adg::DesignBuilder design(store);
  auto system = design.createSystem("hardware-test-system");
  if (!system)
    return system.takeError();
  auto imported = system->importSpatialCore(module);
  if (!imported)
    return imported.takeError();
  auto architecture = adg::getBuiltinInstructionCoreArchitecture();
  if (!architecture)
    return architecture.takeError();
  auto microarchitecture = makeInstructionCoreMicroarchitecture();
  if (!microarchitecture)
    return microarchitecture.takeError();
  auto host = system->addHostCore(*architecture, *microarchitecture);
  if (!host)
    return host.takeError();
  std::vector<adg::HardwareDomainMember> domainMembers{host->domainMember()};
  for (std::uint64_t ordinal = 0; ordinal < spatialCoreCount; ++ordinal) {
    auto core =
        system->addAccCore(*architecture, *microarchitecture, *imported);
    if (!core)
      return core.takeError();
    domainMembers.push_back(core->instructionCoreDomainMember());
    domainMembers.push_back(core->spatialCoreDomainMember());
  }
  auto clock = system->createHardwareDomain();
  if (!clock)
    return clock.takeError();
  auto clockContract = fabric::ClockDomainContractRecord::create(1'000, 0);
  if (!clockContract)
    return clockContract.takeError();
  if (llvm::Error error =
          clock->close(domainMembers, std::move(*clockContract)))
    return std::move(error);
  auto reset = system->createHardwareDomain();
  if (!reset)
    return reset.takeError();
  auto resetContract = fabric::ResetDomainContractRecord::create(
      fabric::ResetPolarity::ActiveHigh, fabric::ResetTiming::Asynchronous,
      fabric::ResetTiming::Asynchronous, fabric::ResetInitialState::Asserted,
      std::nullopt, 0);
  if (!resetContract)
    return resetContract.takeError();
  if (llvm::Error error =
          reset->close(domainMembers, std::move(*resetContract)))
    return std::move(error);
  if (llvm::Error error = system->close())
    return std::move(error);
  auto finalized = std::move(design).finalize();
  if (!finalized)
    return finalized.takeError();
  if (finalized->roots().size() != 1)
    return invalid("test System did not finalize exactly one root");
  return fabric::importEntireFabricRoot(finalized->roots().front().reference(),
                                        store);
}

llvm::Expected<fabric::FinalizedFabricRoot>
makeSingleSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                            const ArtifactStore &store) {
  return makeSpatialCoreSystem(module, store, 1);
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyPhysicalConfigurationField(
    const fabric::FabricPhysicalOccurrenceOwnerRef &owner,
    fabric::FabricOrdinal fieldOrdinal) {
  if (owner.kind() !=
      fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
    return invalid("configuration owner is not Module-internal");
  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(owner.payload());
  if (internal.target.kind() != fabric::FabricModulePhysicalTargetKind::Owner)
    return invalid("configuration owner is not a physical owner target");
  const auto &localOwner =
      std::get<fabric::FabricModulePhysicalOwnerRef>(internal.target.payload());
  const fabric::FabricSemanticConfigFieldRef localField{
      fabric::FabricConfigurationOwnerRef(inventoryOwner(localOwner)),
      fieldOrdinal};
  return qualifyField(internal.spatialCore, localField);
}

llvm::Expected<ConfigurationABIDraft> makeCompleteConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &systemRoot,
    llvm::ArrayRef<ConfigurationFieldEncodingOverride> overrides) {
  auto system = fabric::requireSystemRoot(systemRoot.view());
  if (!system)
    return system.takeError();

  std::map<OverrideKey, const ConfigurationFieldEncodingOverride *>
      overrideByField;
  for (const ConfigurationFieldEncodingOverride &override : overrides)
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
        if (override != overrideByField.end()) {
          fields.push_back(
              ConfigurationFieldEncoding{override->second->field,
                                         override->second->semanticEncoding,
                                         {},
                                         override->second->inactiveValue});
          consumedOverrides.insert(key);
          continue;
        }
        auto field = defaultFieldEncoding(*module, spatialCore, localField);
        if (!field)
          return field.takeError();
        fields.push_back(std::move(*field));
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

} // namespace loom::hardware::test

#include "Mapping/Artifact/ConfiguredHardwareProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace detail {

struct PhysicalConfiguredHardwareProjectionViewAccess final {
  static PhysicalConfiguredHardwareProjectionView
  create(std::vector<PhysicalConfiguredHardwareFieldValueView> fields) {
    return PhysicalConfiguredHardwareProjectionView(std::move(fields));
  }
};

} // namespace detail
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "configured_hardware_projection_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::vector<PhysicalConfiguredHardwareFieldValueView>>
qualifyFields(const ConfiguredHardwareProjectionView &projection,
              ::loom::fabric::SpatialCoreOccurrenceRef occurrence) {
  std::vector<PhysicalConfiguredHardwareFieldValueView> result;
  result.reserve(projection.fields().size());
  for (const ConfiguredHardwareFieldValueView &field : projection.fields()) {
    auto slot = ::loom::fabric::FabricPhysicalConfigurationSlotRef::create(
        ::loom::fabric::SpatialCoreInternalConfigurationSlotRef{occurrence,
                                                                field.slot});
    if (!slot)
      return slot.takeError();
    result.push_back({std::move(*slot), field.value});
  }
  return result;
}

llvm::Expected<std::vector<PhysicalConfiguredHardwareFieldValueView>>
deriveDirectSystemFields(const SystemMappingView &mapping,
                         const ::loom::fabric::FabricSystemRootView &system) {
  std::map<::loom::fabric::FabricEntityId,
           std::vector<::loom::fabric::FabricTransferPatternRef>>
      selectedByResource;
  for (const SystemServiceRealizationView &service :
       mapping.serviceRealizations())
    for (const SystemServicePlanView &plan : service.plans)
      for (const SystemTransferLegView &leg : plan.transferLegs)
        for (const SystemTransferRouteNodeView &node : leg.nodes) {
          const auto *payload =
              std::get_if<::loom::fabric::FabricTransferPatternLegPayload>(
                  &node.incomingTraversal.payload);
          if (payload)
            selectedByResource[payload->owner.resource.id()].push_back(
                payload->owner);
        }

  std::vector<PhysicalConfiguredHardwareFieldValueView> result;
  for (const auto resource : system.transportResources()) {
    const ::loom::fabric::FabricInventoryOwnerRef owner =
        ::loom::fabric::FabricInventoryOwnerRef::of(resource);
    const std::uint64_t fieldCount = system.artifact().inventorySize(
        owner, ::loom::fabric::FabricInventoryKind::SemanticConfigField);
    if (fieldCount == 0)
      continue;
    if (fieldCount != 1)
      return invalid("System transport resource has an unsupported "
                     "configuration field inventory");

    auto &selected = selectedByResource[resource.id()];
    std::sort(selected.begin(), selected.end(),
              [](const auto &left, const auto &right) {
                return left.ordinal < right.ordinal;
              });
    selected.erase(std::unique(selected.begin(), selected.end()),
                   selected.end());
    const ::loom::fabric::FabricSemanticConfigFieldRef field{
        ::loom::fabric::FabricConfigurationOwnerRef(owner), 0};
    auto value = ::loom::fabric::encodeSystemTransportResourceConfiguration(
        system.artifact(), field, selected);
    if (!value)
      return value.takeError();
    auto residencies = system.artifact().configurationResidencies(field);
    if (!residencies)
      return residencies.takeError();
    for (const ::loom::fabric::FabricConfigurationResidency &residency :
         *residencies) {
      auto slot = ::loom::fabric::FabricPhysicalConfigurationSlotRef::create(
          ::loom::fabric::FabricConfigurationSlotRef{field, residency});
      if (!slot)
        return slot.takeError();
      result.push_back({std::move(*slot), *value});
    }
  }
  return result;
}

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
canonicalize(std::vector<PhysicalConfiguredHardwareFieldValueView> values) {
  std::map<ByteVector, PhysicalConfiguredHardwareFieldValueView> bySlot;
  for (PhysicalConfiguredHardwareFieldValueView &value : values) {
    ByteVector key = ::loom::fabric::canonicalFabricBytes(value.slot);
    auto found = bySlot.find(key);
    if (found == bySlot.end()) {
      bySlot.emplace(std::move(key), std::move(value));
      continue;
    }
    if (!found->second.value.bytes().equals(value.value.bytes()))
      return invalid("one occurrence-qualified slot has conflicting values");
  }

  std::vector<PhysicalConfiguredHardwareFieldValueView> ordered;
  ordered.reserve(bySlot.size());
  for (auto &[key, value] : bySlot) {
    (void)key;
    ordered.push_back(std::move(value));
  }
  return detail::PhysicalConfiguredHardwareProjectionViewAccess::create(
      std::move(ordered));
}

} // namespace

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
qualifyConfiguredHardwareProjection(
    const FinalizedSpatialMapping &mapping,
    const ::loom::fabric::FabricSystemRootView &system,
    ::loom::fabric::SpatialCoreOccurrenceRef occurrence) {
  const auto target = system.spatialCoreTarget(occurrence.core);
  if (!target)
    return invalid("SpatialCore occurrence has no imported Module");
  const auto modules = system.artifact().importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalid("SpatialCore occurrence has an invalid Module dependency");
  if (modules[target->dependencyOrdinal].identity() !=
      mapping.view().fabricIdentity())
    return invalid("SpatialMapping does not bind the occurrence's imported "
                   "Module");
  auto values = qualifyFields(mapping.view().configuredHardware(), occurrence);
  if (!values)
    return values.takeError();
  return canonicalize(std::move(*values));
}

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(const FinalizedSystemMapping &mapping,
                                   const ArtifactStore &store) {
  const ArtifactRootReference systemReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      mapping.view().fabricIdentity()};
  auto systemRoot =
      ::loom::fabric::importEntireFabricRoot(systemReference, store);
  if (!systemRoot)
    return systemRoot.takeError();
  auto system = ::loom::fabric::requireSystemRoot(systemRoot->view());
  if (!system)
    return system.takeError();

  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      mapping.view().dataflowIdentity()};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  auto contexts = projectSystemExecutionContexts(
      *dataflowView, mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();

  std::map<std::string, FinalizedSpatialMapping> imported;
  auto direct = deriveDirectSystemFields(mapping.view(), *system);
  if (!direct)
    return direct.takeError();
  std::vector<PhysicalConfiguredHardwareFieldValueView> values =
      std::move(*direct);
  for (const SystemSpatialContextDomain &domain : contexts->spatialDomains) {
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    auto found = imported.find(mappingKey);
    if (found == imported.end()) {
      auto spatial = importSpatialMapping(domain.spatialMapping, store);
      if (!spatial)
        return spatial.takeError();
      found = imported.emplace(mappingKey, std::move(*spatial)).first;
    }
    auto qualified = qualifyFields(
        found->second.view().configuredHardware(),
        ::loom::fabric::SpatialCoreOccurrenceRef{domain.context.accCore});
    if (!qualified)
      return qualified.takeError();
    values.insert(values.end(), std::make_move_iterator(qualified->begin()),
                  std::make_move_iterator(qualified->end()));
  }
  return canonicalize(std::move(values));
}

} // namespace loom::mapping

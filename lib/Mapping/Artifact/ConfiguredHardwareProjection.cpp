#include "ConfiguredHardwareProjectionInternal.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

struct SlotKey final {
  ByteVector slot;

  friend bool operator<(const SlotKey &lhs, const SlotKey &rhs) {
    return lhs.slot < rhs.slot;
  }
};

struct RouteTraversalUse final {
  ::loom::fabric::FabricPhysicalTraversalRef traversal;
  std::uint64_t routeOrdinal = 0;
  std::uint64_t nodeOrdinal = 0;
};

const SpatialComputeBindingView *
findBinding(llvm::ArrayRef<SpatialComputeBindingView> bindings,
            std::uint64_t realization) {
  const SpatialComputeBindingView *result = nullptr;
  for (const SpatialComputeBindingView &binding : bindings) {
    if (binding.realization != realization)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

llvm::Expected<std::optional<::loom::PointerLayout>>
resolvePointerLayout(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto addressSpace = ::dataflow::projectActorPointerAddressSpace(actor);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<::loom::PointerLayout>();
  auto layout = dataflow.pointerLayout(**addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<::loom::PointerLayout>(*layout);
}

SlotKey key(const ::loom::fabric::FabricConfigurationSlotRef &slot) {
  return {::loom::fabric::canonicalFabricBytes(slot)};
}

void appendRouteTraversals(const SpatialRouteTreeView &route,
                           std::uint64_t routeOrdinal,
                           std::vector<RouteTraversalUse> &result) {
  if (route.localTraversal)
    result.push_back({*route.localTraversal, routeOrdinal, 0});
  for (const SpatialRouteNodeView &node : route.nodes)
    if (node.incomingTraversal)
      result.push_back({*node.incomingTraversal, routeOrdinal, node.ordinal});
  for (const SpatialRouteSinkView &sink : route.sinks)
    if (sink.localTraversal)
      result.push_back({*sink.localTraversal, routeOrdinal, sink.nodeOrdinal});
}

} // namespace

llvm::Expected<::loom::fabric::FabricConfigurationSlotRef>
resolveConfiguredHardwareSlot(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricSemanticConfigFieldRef &field,
    std::optional<::loom::fabric::InstructionContextRef> instructionContext) {
  auto residencies = fabric.configurationResidencies(field);
  if (!residencies)
    return residencies.takeError();
  const ::loom::fabric::FabricConfigurationResidency staticResidency =
      ::loom::fabric::FabricStaticConfigurationResidency{};
  if (llvm::is_contained(*residencies, staticResidency))
    return ::loom::fabric::FabricConfigurationSlotRef{field, staticResidency};
  if (!instructionContext)
    return invalid("configuration field requires an instruction context");
  const ::loom::fabric::FabricConfigurationResidency selected =
      *instructionContext;
  if (!llvm::is_contained(*residencies, selected))
    return invalid("configuration field does not admit its bound instruction "
                   "context");
  return ::loom::fabric::FabricConfigurationSlotRef{field, selected};
}

llvm::Expected<ConfiguredHardwareProjectionView>
canonicalizeConfiguredHardwareProjection(
    std::vector<ConfiguredHardwareFieldValueView> selectedFields) {
  std::map<SlotKey, ConfiguredHardwareFieldValueView> fields;
  for (ConfiguredHardwareFieldValueView &selected : selectedFields) {
    const SlotKey selectedKey = key(selected.slot);
    auto found = fields.find(selectedKey);
    if (found == fields.end()) {
      fields.emplace(selectedKey, std::move(selected));
      continue;
    }
    if (!found->second.value.bytes().equals(selected.value.bytes()))
      return invalid("one physical configuration field has conflicting "
                     "semantic values");
  }

  std::vector<ConfiguredHardwareFieldValueView> orderedFields;
  orderedFields.reserve(fields.size());
  for (auto &[slot, value] : fields) {
    (void)slot;
    orderedFields.push_back(std::move(value));
  }
  return ConfiguredHardwareProjectionViewAccess::create(
      std::move(orderedFields));
}

llvm::Expected<ConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryEngines,
    llvm::ArrayRef<SpatialMemoryBindingView> memoryBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView>
        operandQueueMatchGroups) {
  if (bindings.size() != techMapping.computeRealizations().size())
    return invalid("configured hardware projection has incomplete bindings");

  std::vector<ConfiguredHardwareFieldValueView> fields;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    const SpatialComputeBindingView *binding =
        findBinding(bindings, realization.entityId);
    if (!binding)
      return invalid("configured hardware projection has a missing or "
                     "duplicate compute binding");

    const ::loom::fabric::FabricSemanticConfigFieldRef fuField{
        ::loom::fabric::FabricConfigurationOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(binding->occurrence)),
        0};
    auto fuSlot =
        resolveConfiguredHardwareSlot(fabric, fuField, binding->context);
    if (!fuSlot)
      return fuSlot.takeError();
    auto fuValue = ::loom::fabric::encodeFabricFuConfiguration(
        fabric, fuField, realization.capabilityTemplate);
    if (!fuValue)
      return fuValue.takeError();
    fields.push_back({std::move(*fuSlot), std::move(*fuValue)});

    for (const TechComputeActorView &actorBinding : realization.actors) {
      auto actor = dataflow.resolve(actorBinding.actor);
      if (!actor)
        return actor.takeError();
      auto actorProjection =
          ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
      if (!actorProjection)
        return actorProjection.takeError();
      auto indexBitWidth = ::loom::getIndexBitWidth(actor->op);
      if (!indexBitWidth)
        return indexBitWidth.takeError();
      auto pointerLayout = resolvePointerLayout(dataflow, *actorProjection);
      if (!pointerLayout)
        return pointerLayout.takeError();

      auto occurrenceOperation = ::loom::fabric::deriveFabricFuOccurrenceNode(
          fabric, actorBinding.fabricOperation, binding->occurrence);
      if (!occurrenceOperation)
        return occurrenceOperation.takeError();
      const auto *capability =
          fabric.resolvedFabricOpCapability(*occurrenceOperation);
      if (!capability)
        return invalid("configured compute actor has no Fabric capability");

      for (const ::loom::fabric::FabricSemanticConfigFieldRef &templateField :
           capability->configurationFieldSchema) {
        auto value = capability->encodeSemanticConfiguration(
            templateField, *actorProjection, *indexBitWidth,
            actorBinding.operandPorts, actorBinding.resultPorts,
            *pointerLayout ? &**pointerLayout : nullptr);
        if (!value) {
          std::string actorType;
          llvm::raw_string_ostream typeStream(actorType);
          typeStream << actorProjection->type;
          return invalid(
              "configured compute actor semantic encoding failed: actor=" +
              llvm::Twine(actorBinding.actor.entity.value()) +
              ", schema=" +
              ::dataflow::operationSchemaSpelling(actorProjection->schema) +
              ", realization=" + llvm::Twine(realization.entityId) +
              ", field=" + llvm::Twine(templateField.ordinal) +
              ", type=" + typeStream.str() + ": " +
              llvm::toString(value.takeError()));
        }
        const ::loom::fabric::FabricSemanticConfigFieldRef occurrenceField{
            ::loom::fabric::FabricConfigurationOwnerRef(
                ::loom::fabric::FabricInventoryOwnerRef::of(
                    *occurrenceOperation)),
            templateField.ordinal};
        if (llvm::Error error =
                ::loom::fabric::validateFabricRef(fabric, occurrenceField))
          return std::move(error);

        auto slot = resolveConfiguredHardwareSlot(fabric, occurrenceField,
                                                  binding->context);
        if (!slot)
          return slot.takeError();

        fields.push_back({std::move(*slot), std::move(*value)});
      }
    }
  }

  auto peFields = deriveConfiguredPeFields(
      fabric, techMapping, bindings, registerFifoTransfers, routes,
      resourceUses, physicalTagSegments, operandQueueMatchGroups);
  if (!peFields)
    return peFields.takeError();
  fields.insert(fields.end(), std::make_move_iterator(peFields->begin()),
                std::make_move_iterator(peFields->end()));

  auto boundaryFields = deriveConfiguredBoundaryFields(
      fabric, routes, resourceUses, physicalTagSegments);
  if (!boundaryFields)
    return boundaryFields.takeError();
  fields.insert(fields.end(), std::make_move_iterator(boundaryFields->begin()),
                std::make_move_iterator(boundaryFields->end()));

  std::vector<RouteTraversalUse> routeTraversals;
  for (const auto &[routeOrdinal, route] : llvm::enumerate(routes))
    appendRouteTraversals(route, routeOrdinal, routeTraversals);
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
  traversals.reserve(routeTraversals.size());
  for (const RouteTraversalUse &use : routeTraversals)
    traversals.push_back(use.traversal);
  llvm::sort(traversals, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  traversals.erase(std::unique(traversals.begin(), traversals.end()),
                   traversals.end());

  auto temporalSwitchRows = deriveSpatialTemporalSwitchPackedRows(
      fabric, routes, resourceUses, physicalTagSegments);
  if (!temporalSwitchRows)
    return temporalSwitchRows.takeError();

  for (const auto sw : fabric.switchOccurrences()) {
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> selected;
    for (const auto &traversal : traversals) {
      const auto *payload =
          std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
              &traversal.payload);
      if (payload && payload->owner == sw)
        selected.push_back(traversal);
    }
    if (selected.empty())
      continue;
    const ::loom::fabric::FabricSemanticConfigFieldRef field{
        ::loom::fabric::FabricConfigurationOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(sw)),
        0};
    auto slot = resolveConfiguredHardwareSlot(fabric, field);
    if (!slot)
      return slot.takeError();
    const auto encodeSwitch =
        [&]() -> llvm::Expected<::loom::CanonicalSemanticBytes> {
      if (fabric.switchSchedule(sw) != ::fabric::Schedule::Temporal)
        return ::loom::fabric::encodeSpatialSwitchConfiguration(fabric, field,
                                                                selected);
      std::vector<::loom::fabric::FabricTemporalSwitchRouteEntry> entries;
      for (const SpatialTemporalSwitchPackedRowView &row :
           *temporalSwitchRows) {
        if (row.occurrence != sw)
          continue;
        entries.push_back({row.tag, row.traversals});
      }
      return ::loom::fabric::encodeTemporalSwitchConfiguration(fabric, field,
                                                               entries);
    };
    auto value = encodeSwitch();
    if (!value)
      return value.takeError();
    fields.push_back({std::move(*slot), std::move(*value)});
  }

  for (const auto fifo : fabric.fifoOccurrences()) {
    std::optional<::loom::fabric::FabricFifoTraversalMode> selectedMode;
    for (const auto &traversal : traversals) {
      const auto *payload =
          std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
              &traversal.payload);
      if (!payload || payload->owner != fifo)
        continue;
      if (selectedMode && *selectedMode != payload->mode)
        return invalid("one FIFO selects both Buffered and Bypass modes");
      selectedMode = payload->mode;
    }
    if (!selectedMode)
      continue;
    const ::loom::fabric::FabricSemanticConfigFieldRef field{
        ::loom::fabric::FabricConfigurationOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(fifo)),
        0};
    auto slot = resolveConfiguredHardwareSlot(fabric, field);
    if (!slot)
      return slot.takeError();
    auto value = ::loom::fabric::encodeFabricFifoConfiguration(fabric, field,
                                                               selectedMode);
    if (!value)
      return value.takeError();
    fields.push_back({std::move(*slot), std::move(*value)});
  }

  auto memoryFields = deriveConfiguredMemoryFields(
      dataflow, techMapping, fabric, memoryEngines, memoryBindings, routes,
      resourceUses, physicalTagSegments);
  if (!memoryFields)
    return memoryFields.takeError();
  fields.insert(fields.end(), std::make_move_iterator(memoryFields->begin()),
                std::make_move_iterator(memoryFields->end()));
  return canonicalizeConfiguredHardwareProjection(std::move(fields));
}

} // namespace loom::mapping::detail

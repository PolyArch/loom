#include "ConfiguredHardwareProjectionInternal.h"

#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

struct PeBindingGroup final {
  ::loom::fabric::FabricPeOccurrenceRef pe;
  std::vector<const SpatialComputeBindingView *> bindings;
};

struct PeSelectorUse final {
  ::loom::fabric::FabricPeSelectorPayload selector;
  std::uint64_t routeOrdinal = 0;
  std::uint64_t nodeOrdinal = 0;
};

void appendSelectorUse(
    const std::optional<::loom::fabric::FabricPhysicalTraversalRef> &traversal,
    std::uint64_t routeOrdinal, std::uint64_t nodeOrdinal,
    std::vector<PeSelectorUse> &uses) {
  if (!traversal)
    return;
  const auto *selector =
      std::get_if<::loom::fabric::FabricPeSelectorPayload>(&traversal->payload);
  if (selector)
    uses.push_back({*selector, routeOrdinal, nodeOrdinal});
}

std::vector<PeSelectorUse>
collectSelectorUses(llvm::ArrayRef<SpatialRouteTreeView> routes) {
  std::vector<PeSelectorUse> result;
  for (auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    appendSelectorUse(route.localTraversal, routeOrdinal, 0, result);
    for (const SpatialRouteSinkView &sink : route.sinks)
      appendSelectorUse(sink.localTraversal, routeOrdinal, sink.nodeOrdinal,
                        result);
  }
  return result;
}

llvm::Expected<std::vector<PeBindingGroup>>
groupBindings(const ::loom::fabric::FabricArtifactView &fabric,
              llvm::ArrayRef<SpatialComputeBindingView> bindings) {
  std::map<ByteVector, PeBindingGroup> groups;
  for (const SpatialComputeBindingView &binding : bindings) {
    const auto pe = fabric.parentPeOf(binding.occurrence);
    if (!pe)
      return invalid("configured FU occurrence has no parent PE");
    const ByteVector key = ::loom::fabric::canonicalFabricBytes(*pe);
    auto [found, inserted] = groups.try_emplace(key, PeBindingGroup{*pe, {}});
    (void)inserted;
    found->second.bindings.push_back(&binding);
  }
  std::vector<PeBindingGroup> result;
  result.reserve(groups.size());
  for (auto &[key, group] : groups) {
    (void)key;
    result.push_back(std::move(group));
  }
  return result;
}

llvm::Expected<std::optional<PeSelectorUse>>
findSelectorUse(const ::loom::fabric::FabricArtifactView &fabric,
                llvm::ArrayRef<PeSelectorUse> uses,
                const ::loom::fabric::FabricFuOccurrencePortRef &port) {
  const auto pe = fabric.parentPeOf(port.fu);
  const auto fixed = fabric.fuOccurrenceTransportEndpoint(port);
  if (!pe || !fixed)
    return invalid("configured FU port has no exact PE attachment endpoint");

  std::optional<PeSelectorUse> result;
  for (const PeSelectorUse &use : uses) {
    if (use.selector.owner != *pe)
      continue;
    const bool matches =
        port.direction == ::loom::fabric::FabricPortDirection::Input
            ? use.selector.destination == *fixed
            : use.selector.source == *fixed;
    if (!matches)
      continue;
    if (result)
      return invalid("one active FU port selects multiple PE boundary routes");
    result = use;
  }
  return result;
}

llvm::Expected<::loom::fabric::FabricTransportEndpointRef>
peEndpointFor(const PeSelectorUse &use,
              ::loom::fabric::FabricPortDirection direction) {
  const auto endpoint = direction == ::loom::fabric::FabricPortDirection::Input
                            ? use.selector.source
                            : use.selector.destination;
  const auto owner =
      ::loom::fabric::FabricTransportEndpointOwnerRef::of(use.selector.owner);
  if (endpoint.owner != owner)
    return invalid("PE selector route has a foreign boundary endpoint");
  return endpoint;
}

llvm::Expected<::loom::fabric::FabricOrdinal>
directionOrdinal(const ::loom::fabric::FabricArtifactView &fabric,
                 const ::loom::fabric::FabricTransportEndpointRef &endpoint,
                 ::loom::fabric::FabricPortDirection direction) {
  const auto actualDirection = fabric.transportEndpointDirection(endpoint);
  if (!actualDirection || *actualDirection != direction)
    return invalid("PE selector boundary endpoint has the wrong direction");
  ::loom::fabric::FabricOrdinal result = 0;
  for (::loom::fabric::FabricOrdinal ordinal = 0; ordinal < endpoint.ordinal;
       ++ordinal) {
    const ::loom::fabric::FabricTransportEndpointRef candidate{endpoint.owner,
                                                               ordinal};
    if (fabric.transportEndpointDirection(candidate) == direction)
      ++result;
  }
  return result;
}

llvm::Error
appendSpatialPeFields(const ::loom::fabric::FabricArtifactView &fabric,
                      const PeBindingGroup &group,
                      llvm::ArrayRef<PeSelectorUse> selectorUses,
                      std::vector<ConfiguredHardwareFieldValueView> &fields) {
  if (group.bindings.size() != 1)
    return invalid("one Spatial PE selects multiple compute realizations");
  const SpatialComputeBindingView &binding = *group.bindings.front();
  auto schema = fabric.spatialPeConfigurationSchema(group.pe);
  if (!schema)
    return schema.takeError();

  for (const ::loom::fabric::FabricPeConfigurationFieldView &field :
       schema->fields()) {
    std::optional<::loom::fabric::FabricPeConfigurationValue> selected;
    if (field.kind ==
        ::loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      selected = ::loom::fabric::FabricPeActive{binding.occurrence};
    } else if (field.port && field.port->fu == binding.occurrence) {
      auto use = findSelectorUse(fabric, selectorUses, *field.port);
      if (!use)
        return use.takeError();
      if (*use) {
        auto endpoint = peEndpointFor(**use, field.port->direction);
        if (!endpoint)
          return endpoint.takeError();
        selected = ::loom::fabric::FabricPeRoute{*endpoint};
      } else {
        selected = ::loom::fabric::FabricPeDisconnected{};
      }
    }
    if (!selected)
      continue;

    auto slot = resolveConfiguredHardwareSlot(fabric, field.reference);
    if (!slot)
      return slot.takeError();
    auto value = schema->encode(field.reference, *selected);
    if (!value)
      return value.takeError();
    fields.push_back({std::move(*slot), std::move(*value)});
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::APInt>
selectorTag(const ::loom::fabric::FabricArtifactView &fabric,
            llvm::ArrayRef<SpatialRouteTreeView> routes,
            llvm::ArrayRef<SpatialResourceUseView> resourceUses,
            llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
            const PeSelectorUse &use, std::uint32_t expectedWidth) {
  auto tag = resolveConfiguredHardwarePhysicalTag(fabric, routes, resourceUses,
                                                  tagSegments, use.routeOrdinal,
                                                  use.nodeOrdinal);
  if (!tag)
    return tag.takeError();
  if (tag->getBitWidth() != expectedWidth)
    return invalid("Temporal PE selector tag has the wrong width");
  return std::move(*tag);
}

llvm::Expected<::loom::fabric::FabricTemporalPeOperandSelection>
temporalOperandSelection(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
    llvm::ArrayRef<PeSelectorUse> selectorUses,
    const ::loom::fabric::FabricFuOccurrencePortRef &port,
    std::uint32_t tagWidth) {
  auto use = findSelectorUse(fabric, selectorUses, port);
  if (!use)
    return use.takeError();
  if (!*use)
    return ::loom::fabric::FabricTemporalPeOperandSelection{
        ::loom::fabric::FabricTemporalPeSelectorKind::Disconnected,
        std::nullopt, llvm::APInt(tagWidth, 0)};
  auto endpoint = peEndpointFor(**use, port.direction);
  if (!endpoint)
    return endpoint.takeError();
  auto target = directionOrdinal(fabric, *endpoint, port.direction);
  if (!target)
    return target.takeError();
  auto tag =
      selectorTag(fabric, routes, resourceUses, tagSegments, **use, tagWidth);
  if (!tag)
    return tag.takeError();
  return ::loom::fabric::FabricTemporalPeOperandSelection{
      ::loom::fabric::FabricTemporalPeSelectorKind::Route,
      ::loom::fabric::FabricTemporalPeSelectorTarget(
          ::loom::fabric::FabricTemporalPePortTarget{*target}),
      std::move(*tag)};
}

llvm::Expected<::loom::fabric::FabricTemporalPeResultSelection>
temporalResultSelection(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
    llvm::ArrayRef<PeSelectorUse> selectorUses,
    const ::loom::fabric::FabricFuOccurrencePortRef &port,
    std::uint32_t tagWidth) {
  auto use = findSelectorUse(fabric, selectorUses, port);
  if (!use)
    return use.takeError();
  if (!*use)
    return ::loom::fabric::FabricTemporalPeResultSelection{
        ::loom::fabric::FabricTemporalPeSelectorKind::Disconnected,
        std::nullopt, llvm::APInt(tagWidth, 0)};
  auto endpoint = peEndpointFor(**use, port.direction);
  if (!endpoint)
    return endpoint.takeError();
  auto target = directionOrdinal(fabric, *endpoint, port.direction);
  if (!target)
    return target.takeError();
  auto tag =
      selectorTag(fabric, routes, resourceUses, tagSegments, **use, tagWidth);
  if (!tag)
    return tag.takeError();
  return ::loom::fabric::FabricTemporalPeResultSelection{
      ::loom::fabric::FabricTemporalPeSelectorKind::Route,
      ::loom::fabric::FabricTemporalPeSelectorTarget(
          ::loom::fabric::FabricTemporalPePortTarget{*target}),
      std::move(*tag)};
}

llvm::Error
appendTemporalPeField(const ::loom::fabric::FabricArtifactView &fabric,
                      const PeBindingGroup &group,
                      llvm::ArrayRef<SpatialRouteTreeView> routes,
                      llvm::ArrayRef<SpatialResourceUseView> resourceUses,
                      llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
                      llvm::ArrayRef<PeSelectorUse> selectorUses,
                      std::vector<ConfiguredHardwareFieldValueView> &fields) {
  auto schema = fabric.temporalPeConfigurationSchema(group.pe);
  if (!schema)
    return schema.takeError();
  const auto &layout = schema->layout();
  ::loom::fabric::FabricTemporalPeActive active;
  active.rows.resize(layout.contextCount);

  for (const SpatialComputeBindingView *binding : group.bindings) {
    if (binding->context.ordinal >= active.rows.size())
      return invalid("Temporal PE binding context is outside instruction_mem");
    if (active.rows[binding->context.ordinal])
      return invalid("one Temporal PE context selects multiple realizations");
    const auto shape = llvm::find_if(layout.fus, [&](const auto &candidate) {
      return candidate.fu == binding->occurrence;
    });
    if (shape == layout.fus.end())
      return invalid("Temporal PE binding selects a foreign FU");

    ::loom::fabric::FabricTemporalPeInstructionEntry row;
    row.selectedFu = binding->occurrence;
    row.operandSelections.reserve(shape->inputCount);
    for (std::uint32_t input = 0; input < shape->inputCount; ++input) {
      auto selection = temporalOperandSelection(
          fabric, routes, resourceUses, tagSegments, selectorUses,
          {binding->occurrence, ::loom::fabric::FabricPortDirection::Input,
           input},
          layout.tagWidthBits);
      if (!selection)
        return selection.takeError();
      row.operandSelections.push_back(std::move(*selection));
    }
    row.resultSelections.reserve(shape->outputCount);
    for (std::uint32_t output = 0; output < shape->outputCount; ++output) {
      auto selection = temporalResultSelection(
          fabric, routes, resourceUses, tagSegments, selectorUses,
          {binding->occurrence, ::loom::fabric::FabricPortDirection::Output,
           output},
          layout.tagWidthBits);
      if (!selection)
        return selection.takeError();
      row.resultSelections.push_back(std::move(*selection));
    }
    active.rows[binding->context.ordinal] = std::move(row);
  }

  auto slot = resolveConfiguredHardwareSlot(fabric, schema->field());
  if (!slot)
    return slot.takeError();
  auto value = schema->encode(
      ::loom::fabric::FabricTemporalPeConfigurationValue{std::move(active)});
  if (!value)
    return value.takeError();
  fields.push_back({std::move(*slot), std::move(*value)});
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredPeFields(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  auto groups = groupBindings(fabric, bindings);
  if (!groups)
    return groups.takeError();
  const std::vector<PeSelectorUse> selectorUses = collectSelectorUses(routes);
  std::vector<ConfiguredHardwareFieldValueView> fields;
  for (const PeBindingGroup &group : *groups) {
    const auto schedule = fabric.peSchedule(group.pe);
    if (schedule == ::fabric::Schedule::Spatial) {
      if (llvm::Error error =
              appendSpatialPeFields(fabric, group, selectorUses, fields))
        return std::move(error);
      continue;
    }
    if (schedule != ::fabric::Schedule::Temporal)
      return invalid("configured PE has an unknown schedule");
    if (llvm::Error error =
            appendTemporalPeField(fabric, group, routes, resourceUses,
                                  physicalTagSegments, selectorUses, fields))
      return std::move(error);
  }
  return fields;
}

} // namespace loom::mapping::detail

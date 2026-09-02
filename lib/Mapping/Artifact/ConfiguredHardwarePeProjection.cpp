#include "ConfiguredHardwareProjectionInternal.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
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
  std::uint64_t realization = 0;
  std::uint64_t routeOrdinal = 0;
  std::uint64_t nodeOrdinal = 0;
};

using ActorRealizationMap = std::map<std::uint64_t, std::uint64_t>;

/// FU boundary outputs driven by each compute realization's selected
/// capability template, keyed by realization. A driven output without a route
/// or register-FIFO transfer carries a dead actor result that the PE must drain
/// with an output `Discard`; an undriven output stays `Disconnected`.
using DrivenOutputSet = std::set<::loom::fabric::FabricOrdinal>;
using DrivenOutputMap = std::map<std::uint64_t, DrivenOutputSet>;

llvm::Expected<DrivenOutputMap>
collectDrivenOutputs(const ::loom::fabric::FabricArtifactView &fabric,
                     const TechMappingView &techMapping) {
  DrivenOutputMap result;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    const auto templates =
        fabric.fuCapabilityTemplates(realization.capabilityTemplate.fu);
    if (realization.capabilityTemplate.ordinal >= templates.size())
      return invalid("compute realization selects an absent capability "
                     "template");
    auto edges = ::loom::fabric::projectFabricFuCapabilityTemplateTerminalEdges(
        templates[realization.capabilityTemplate.ordinal]);
    if (!edges)
      return edges.takeError();
    DrivenOutputSet &driven = result[realization.entityId];
    for (const ::loom::fabric::FabricFuCapabilityTemplateEdge &edge : *edges) {
      if (edge.destination.kind() !=
          ::loom::fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
        continue;
      const auto &port = std::get<::loom::fabric::FabricFuTemplatePortRef>(
          edge.destination.payload);
      if (port.direction == ::loom::fabric::FabricPortDirection::Output)
        driven.insert(port.ordinal);
    }
  }
  return result;
}

const DrivenOutputSet &drivenOutputsOf(const DrivenOutputMap &driven,
                                       std::uint64_t realization) {
  static const DrivenOutputSet empty;
  const auto found = driven.find(realization);
  return found == driven.end() ? empty : found->second;
}

void appendSelectorUse(
    const std::optional<::loom::fabric::FabricPhysicalTraversalRef> &traversal,
    std::uint64_t realization, std::uint64_t routeOrdinal,
    std::uint64_t nodeOrdinal, std::vector<PeSelectorUse> &uses) {
  if (!traversal)
    return;
  const auto *selector =
      std::get_if<::loom::fabric::FabricPeSelectorPayload>(&traversal->payload);
  if (selector)
    uses.push_back({*selector, realization, routeOrdinal, nodeOrdinal});
}

llvm::Expected<std::uint64_t>
realizationOf(const ActorRealizationMap &realizations,
              const ::dataflow::ActorRef &actor) {
  const auto found = realizations.find(actor.entity.value());
  if (found == realizations.end())
    return invalid("PE selector terminal has no compute realization");
  return found->second;
}

llvm::Expected<std::vector<PeSelectorUse>>
collectSelectorUses(const TechMappingView &techMapping,
                    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  ActorRealizationMap realizationByActor;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations())
    for (const TechComputeActorView &actor : realization.actors)
      if (!realizationByActor
               .try_emplace(actor.actor.entity.value(), realization.entityId)
               .second)
        return invalid("compute actor belongs to multiple realizations");

  std::vector<PeSelectorUse> result;
  for (auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    if (route.localTraversal) {
      const auto *producer =
          std::get_if<::dataflow::ActorTokenResultRef>(&route.logicalNet);
      if (!producer)
        return invalid("graph boundary producer selects a PE traversal");
      auto realization = realizationOf(realizationByActor, producer->actor);
      if (!realization)
        return realization.takeError();
      appendSelectorUse(route.localTraversal, *realization, routeOrdinal, 0,
                        result);
    }
    for (const SpatialRouteSinkView &sink : route.sinks) {
      if (!sink.localTraversal)
        continue;
      const auto *consumer =
          std::get_if<::dataflow::ActorTokenOperandRef>(&sink.sink);
      if (!consumer)
        return invalid("graph boundary consumer selects a PE traversal");
      auto realization = realizationOf(realizationByActor, consumer->actor);
      if (!realization)
        return realization.takeError();
      appendSelectorUse(sink.localTraversal, *realization, routeOrdinal,
                        sink.nodeOrdinal, result);
    }
  }
  return result;
}

llvm::Expected<std::vector<SpatialPeLocalTransferUse>>
deriveSpatialPeLocalTransferUsesImpl(
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> transfers) {
  ActorRealizationMap realizationByActor;
  std::map<std::uint64_t, const TechComputeRealizationView *> realizations;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    realizations.emplace(realization.entityId, &realization);
    for (const TechComputeActorView &actor : realization.actors)
      if (!realizationByActor
               .try_emplace(actor.actor.entity.value(), realization.entityId)
               .second)
        return invalid("compute actor belongs to multiple realizations");
  }
  const auto bindingOf =
      [&](std::uint64_t realization) -> const SpatialComputeBindingView * {
    const SpatialComputeBindingView *result = nullptr;
    for (const SpatialComputeBindingView &binding : bindings) {
      if (binding.realization != realization)
        continue;
      if (result)
        return nullptr;
      result = &binding;
    }
    return result;
  };
  const auto boundaryOf =
      [&](const TechComputeRealizationView &realization,
          ::dataflow::ActorRef actor,
          ::loom::fabric::FabricPortDirection direction,
          std::uint64_t ordinal) -> const TechComputeBoundaryView * {
    const TechComputeBoundaryView *result = nullptr;
    for (const TechComputeBoundaryView &boundary : realization.boundaries) {
      if (boundary.actor != actor || boundary.direction != direction ||
          boundary.portOrdinal != ordinal)
        continue;
      if (result)
        return nullptr;
      result = &boundary;
    }
    return result;
  };

  std::vector<SpatialPeLocalTransferUse> result;
  result.reserve(transfers.size());
  for (const SpatialRegisterFifoTransferView &transfer : transfers) {
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&transfer.logicalNet);
    const auto *consumer =
        std::get_if<::dataflow::ActorTokenOperandRef>(&transfer.sink);
    if (!producer || !consumer)
      return invalid("register-FIFO transfer has a graph-boundary terminal");
    auto producerRealization =
        realizationOf(realizationByActor, producer->actor);
    auto consumerRealization =
        realizationOf(realizationByActor, consumer->actor);
    if (!producerRealization)
      return producerRealization.takeError();
    if (!consumerRealization)
      return consumerRealization.takeError();
    const auto producerRecord = realizations.find(*producerRealization);
    const auto consumerRecord = realizations.find(*consumerRealization);
    const SpatialComputeBindingView *producerBinding =
        bindingOf(*producerRealization);
    const SpatialComputeBindingView *consumerBinding =
        bindingOf(*consumerRealization);
    if (producerRecord == realizations.end() ||
        consumerRecord == realizations.end() || !producerBinding ||
        !consumerBinding)
      return invalid("register-FIFO transfer has no unique compute binding");
    const TechComputeBoundaryView *producerBoundary = boundaryOf(
        *producerRecord->second, producer->actor,
        ::loom::fabric::FabricPortDirection::Output, producer->ordinal);
    const TechComputeBoundaryView *consumerBoundary = boundaryOf(
        *consumerRecord->second, consumer->actor,
        ::loom::fabric::FabricPortDirection::Input, consumer->ordinal);
    if (!producerBoundary || !consumerBoundary)
      return invalid("register-FIFO transfer has no unique FU boundary");
    const auto producerPe = fabric.parentPeOf(producerBinding->occurrence);
    const auto consumerPe = fabric.parentPeOf(consumerBinding->occurrence);
    if (!producerPe || !consumerPe || *producerPe != transfer.pe ||
        *consumerPe != transfer.pe)
      return invalid("register-FIFO transfer spans physical PEs");
    result.push_back(
        SpatialPeLocalTransferUse{*producerRealization,
                                  *consumerRealization,
                                  {producerBinding->occurrence,
                                   ::loom::fabric::FabricPortDirection::Output,
                                   producerBoundary->fabricPort.ordinal},
                                  {consumerBinding->occurrence,
                                   ::loom::fabric::FabricPortDirection::Input,
                                   consumerBoundary->fabricPort.ordinal},
                                  transfer.pe,
                                  transfer.registerFifo,
                                  transfer.writeTraversal,
                                  transfer.readTraversal,
                                  transfer.tag});
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
                const ::loom::fabric::FabricFuOccurrencePortRef &port,
                std::uint64_t realization) {
  const auto pe = fabric.parentPeOf(port.fu);
  const auto fixed = fabric.fuOccurrenceTransportEndpoint(port);
  if (!pe || !fixed)
    return invalid("configured FU port has no exact PE attachment endpoint");

  std::optional<PeSelectorUse> result;
  for (const PeSelectorUse &use : uses) {
    if (use.realization != realization || use.selector.owner != *pe)
      continue;
    const bool matches =
        port.direction == ::loom::fabric::FabricPortDirection::Input
            ? use.selector.destination == *fixed
            : use.selector.source == *fixed;
    if (!matches)
      continue;
    if (result && result->selector == use.selector &&
        result->routeOrdinal == use.routeOrdinal &&
        result->nodeOrdinal == use.nodeOrdinal)
      continue;
    if (result)
      return invalid("one active FU port selects multiple PE boundary routes "
                     "at route " +
                     llvm::Twine(result->routeOrdinal) + " node " +
                     llvm::Twine(result->nodeOrdinal) + " and route " +
                     llvm::Twine(use.routeOrdinal) + " node " +
                     llvm::Twine(use.nodeOrdinal));
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
                      const DrivenOutputMap &drivenOutputs,
                      std::vector<ConfiguredHardwareFieldValueView> &fields) {
  if (group.bindings.size() != 1)
    return invalid("one Spatial PE selects multiple compute realizations");
  const SpatialComputeBindingView &binding = *group.bindings.front();
  auto schema = fabric.spatialPeConfigurationSchema(group.pe);
  if (!schema)
    return schema.takeError();
  const DrivenOutputSet &driven =
      drivenOutputsOf(drivenOutputs, binding.realization);

  for (const ::loom::fabric::FabricPeConfigurationFieldView &field :
       schema->fields()) {
    std::optional<::loom::fabric::FabricPeConfigurationValue> selected;
    if (field.kind ==
        ::loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      selected = ::loom::fabric::FabricPeActive{binding.occurrence};
    } else if (field.port && field.port->fu == binding.occurrence) {
      auto use = findSelectorUse(fabric, selectorUses, *field.port,
                                 binding.realization);
      if (!use)
        return use.takeError();
      if (*use) {
        auto endpoint = peEndpointFor(**use, field.port->direction);
        if (!endpoint)
          return endpoint.takeError();
        selected = ::loom::fabric::FabricPeRoute{*endpoint};
      } else if (field.port->direction ==
                     ::loom::fabric::FabricPortDirection::Output &&
                 driven.count(field.port->ordinal)) {
        selected = ::loom::fabric::FabricPeOutputDiscard{};
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
    llvm::ArrayRef<SpatialPeLocalTransferUse> localTransferUses,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> matchGroups,
    const ::loom::fabric::FabricFuOccurrencePortRef &port,
    std::uint64_t realization, const ::fabric::LogicalOperandQueueKey &queue,
    std::uint32_t tagWidth) {
  const SpatialPeLocalTransferUse *local = nullptr;
  for (const SpatialPeLocalTransferUse &candidate : localTransferUses) {
    if (candidate.consumerRealization != realization ||
        candidate.consumerPort != port)
      continue;
    if (local)
      return invalid("Temporal PE input selects multiple register FIFOs");
    local = &candidate;
  }
  const SpatialPeOperandQueueMatchGroupView *selected = nullptr;
  for (const SpatialPeOperandQueueMatchGroupView &group : matchGroups) {
    if (!llvm::any_of(group.matches,
                      [&](const auto &match) { return match.queue == queue; }))
      continue;
    if (selected)
      return invalid("Temporal PE operand queue belongs to multiple match "
                     "groups");
    selected = &group;
  }
  if (local) {
    if (selected)
      return invalid("Temporal PE input selects a register FIFO and an "
                     "external operand queue");
    if (local->tag.getBitWidth() != tagWidth)
      return invalid("Temporal PE register-FIFO input has the wrong tag "
                     "width");
    return ::loom::fabric::FabricTemporalPeOperandSelection{
        ::loom::fabric::FabricTemporalPeSelectorKind::Route,
        ::loom::fabric::FabricTemporalPeSelectorTarget(
            ::loom::fabric::FabricTemporalPeRegisterFifoTarget{
                local->registerFifo}),
        local->tag};
  }
  if (!selected)
    return ::loom::fabric::FabricTemporalPeOperandSelection{
        ::loom::fabric::FabricTemporalPeSelectorKind::Disconnected,
        std::nullopt, llvm::APInt(tagWidth, 0)};
  if (selected->tag.getBitWidth() != tagWidth)
    return invalid("Temporal PE operand match group has the wrong tag width");
  auto target = directionOrdinal(fabric, selected->ingress,
                                 ::loom::fabric::FabricPortDirection::Input);
  if (!target)
    return target.takeError();
  return ::loom::fabric::FabricTemporalPeOperandSelection{
      ::loom::fabric::FabricTemporalPeSelectorKind::Route,
      ::loom::fabric::FabricTemporalPeSelectorTarget(
          ::loom::fabric::FabricTemporalPePortTarget{*target}),
      selected->tag};
}

llvm::Expected<::loom::fabric::FabricTemporalPeResultSelection>
temporalResultSelection(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialPeLocalTransferUse> localTransferUses,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
    llvm::ArrayRef<PeSelectorUse> selectorUses,
    const DrivenOutputSet &drivenOutputs,
    const ::loom::fabric::FabricFuOccurrencePortRef &port,
    std::uint64_t realization, std::uint32_t tagWidth) {
  const SpatialPeLocalTransferUse *local = nullptr;
  for (const SpatialPeLocalTransferUse &candidate : localTransferUses) {
    if (candidate.producerRealization != realization ||
        candidate.producerPort != port)
      continue;
    if (local)
      return invalid("Temporal PE output selects multiple register FIFOs");
    local = &candidate;
  }
  auto use = findSelectorUse(fabric, selectorUses, port, realization);
  if (!use)
    return use.takeError();
  if (local) {
    if (*use)
      return invalid("Temporal PE output selects a register FIFO and an "
                     "external route");
    if (local->tag.getBitWidth() != tagWidth)
      return invalid("Temporal PE register-FIFO output has the wrong tag "
                     "width");
    return ::loom::fabric::FabricTemporalPeResultSelection{
        ::loom::fabric::FabricTemporalPeSelectorKind::Route,
        ::loom::fabric::FabricTemporalPeSelectorTarget(
            ::loom::fabric::FabricTemporalPeRegisterFifoTarget{
                local->registerFifo}),
        local->tag};
  }
  if (!*use)
    return ::loom::fabric::FabricTemporalPeResultSelection{
        drivenOutputs.count(port.ordinal)
            ? ::loom::fabric::FabricTemporalPeSelectorKind::Discard
            : ::loom::fabric::FabricTemporalPeSelectorKind::Disconnected,
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

llvm::Error appendTemporalPeField(
    const ::loom::fabric::FabricArtifactView &fabric,
    const PeBindingGroup &group,
    llvm::ArrayRef<SpatialPeLocalTransferUse> localTransferUses,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
    llvm::ArrayRef<PeSelectorUse> selectorUses,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueMatchGroups,
    const DrivenOutputMap &drivenOutputs,
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
    const ::loom::fabric::FabricOrdinal fuOrdinal =
        static_cast<::loom::fabric::FabricOrdinal>(
            std::distance(layout.fus.begin(), shape));

    ::loom::fabric::FabricTemporalPeInstructionEntry row;
    row.selectedFu = binding->occurrence;
    row.operandSelections.reserve(shape->inputCount);
    for (std::uint32_t input = 0; input < shape->inputCount; ++input) {
      auto selection = temporalOperandSelection(
          fabric, localTransferUses, operandQueueMatchGroups,
          {binding->occurrence, ::loom::fabric::FabricPortDirection::Input,
           input},
          binding->realization,
          ::fabric::LogicalOperandQueueKey{binding->context, fuOrdinal, input},
          layout.tagWidthBits);
      if (!selection)
        return selection.takeError();
      row.operandSelections.push_back(std::move(*selection));
    }
    row.resultSelections.reserve(shape->outputCount);
    for (std::uint32_t output = 0; output < shape->outputCount; ++output) {
      auto selection = temporalResultSelection(
          fabric, localTransferUses, routes, resourceUses, tagSegments,
          selectorUses, drivenOutputsOf(drivenOutputs, binding->realization),
          {binding->occurrence, ::loom::fabric::FabricPortDirection::Output,
           output},
          binding->realization, layout.tagWidthBits);
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

llvm::Expected<std::vector<SpatialPeLocalTransferUse>>
deriveSpatialPeLocalTransferUses(
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> transfers) {
  return deriveSpatialPeLocalTransferUsesImpl(fabric, techMapping, bindings,
                                              transfers);
}

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredPeFields(
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView>
        operandQueueMatchGroups) {
  auto groups = groupBindings(fabric, bindings);
  if (!groups)
    return groups.takeError();
  auto selectorUses = collectSelectorUses(techMapping, routes);
  if (!selectorUses)
    return selectorUses.takeError();
  auto localTransferUses = deriveSpatialPeLocalTransferUses(
      fabric, techMapping, bindings, registerFifoTransfers);
  if (!localTransferUses)
    return localTransferUses.takeError();
  auto drivenOutputs = collectDrivenOutputs(fabric, techMapping);
  if (!drivenOutputs)
    return drivenOutputs.takeError();
  std::vector<ConfiguredHardwareFieldValueView> fields;
  for (const PeBindingGroup &group : *groups) {
    const auto schedule = fabric.peSchedule(group.pe);
    if (schedule == ::fabric::Schedule::Spatial) {
      if (llvm::Error error = appendSpatialPeFields(
              fabric, group, *selectorUses, *drivenOutputs, fields))
        return std::move(error);
      continue;
    }
    if (schedule != ::fabric::Schedule::Temporal)
      return invalid("configured PE has an unknown schedule");
    if (llvm::Error error = appendTemporalPeField(
            fabric, group, *localTransferUses, routes, resourceUses,
            physicalTagSegments, *selectorUses, operandQueueMatchGroups,
            *drivenOutputs, fields))
      return std::move(error);
  }
  return fields;
}

} // namespace loom::mapping::detail

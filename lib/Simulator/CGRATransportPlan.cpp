#include "CGRATransportPlan.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

template <typename Size>
llvm::Expected<std::uint32_t> checkedU32(Size value, llvm::StringRef label) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        label + " exceeds u32");
  return static_cast<std::uint32_t>(value);
}

llvm::Expected<::dataflow::GraphRef>
resolveGraphOf(const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  if (const auto *result =
          std::get_if<::dataflow::ActorTokenResultRef>(&producer)) {
    auto actor = dataflow.resolve(result->actor);
    if (!actor)
      return actor.takeError();
    return actor->graph;
  }
  const auto &ingress = std::get<::dataflow::GraphIngressTokenRef>(producer);
  return std::visit([](const auto &typed) { return typed.graph; }, ingress);
}

using RefBytes = std::vector<std::uint8_t>;
using EdgeKey = std::pair<RefBytes, RefBytes>;

RefBytes bytes(const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  return ::loom::fabric::canonicalFabricBytes(reference);
}

template <typename Ref>
llvm::Expected<RefBytes>
dataflowBytes(const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const Ref &reference) {
  return ::dataflow::encodeDataflowReference(dataflow.identity(), reference);
}

void collect(
    const std::optional<::loom::fabric::FabricPhysicalTraversalRef> &reference,
    std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef> &selected) {
  if (reference)
    selected.try_emplace(bytes(*reference), *reference);
}

struct StorageKey final {
  std::uint32_t kind = 0;
  RefBytes owner;
  std::uint64_t ordinal = 0;

  bool operator<(const StorageKey &other) const {
    return std::tie(kind, owner, ordinal) <
           std::tie(other.kind, other.owner, other.ordinal);
  }
};

enum class TraversalActivationKeyKind : std::uint8_t {
  UsePatternRequester,
  SpatialSwitchInput,
};

struct TraversalActivationKey final {
  TraversalActivationKeyKind kind =
      TraversalActivationKeyKind::UsePatternRequester;
  RefBytes owner;
  ::loom::fabric::FabricOrdinal ordinal = 0;

  bool operator<(const TraversalActivationKey &other) const {
    return std::tie(kind, owner, ordinal) <
           std::tie(other.kind, other.owner, other.ordinal);
  }
};

llvm::Expected<TraversalActivationKey> traversalActivationKey(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTraversalRef &traversal,
    const ::loom::fabric::FabricTraversalUseView &use) {
  if (const auto *sw =
          std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
              &traversal.payload)) {
    if (fabric.switchSchedule(sw->owner) != ::fabric::Schedule::Spatial)
      return invalid("Temporal switch activation bypassed packed-row "
                     "projection");
    if (use.requesterGroup.kind !=
            ::loom::fabric::FabricTraversalRequesterGroupKind::
                SwitchRequester ||
        use.requesterGroup.owner !=
            ::loom::fabric::FabricInventoryOwnerRef::of(sw->owner))
      return invalid("Spatial switch traversal has a foreign requester");
    return TraversalActivationKey{
        TraversalActivationKeyKind::SpatialSwitchInput,
        ::loom::fabric::canonicalFabricBytes(sw->owner), sw->input};
  }
  if (use.requesterGroup.kind !=
      ::loom::fabric::FabricTraversalRequesterGroupKind::UsePattern)
    return invalid("ordinary traversal has a switch requester");
  return TraversalActivationKey{
      TraversalActivationKeyKind::UsePatternRequester,
      ::loom::fabric::canonicalFabricBytes(use.requesterGroup.owner),
      use.requesterGroup.ordinal};
}

struct StorageProjection final {
  CgraTraversalStorageKind accessKind = CgraTraversalStorageKind::None;
  std::uint32_t capacity = 0;
  StorageKey key;
  ::loom::fabric::FabricUsePatternRef enqueuePattern;
  ::loom::fabric::FabricUsePatternRef dequeuePattern;
  std::optional<::loom::fabric::FabricUsePatternRef> simultaneousPattern;
};

llvm::Expected<std::optional<StorageProjection>>
storageContract(const ::loom::fabric::FabricArtifactView &fabric,
                const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  if (const auto *fifo =
          std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
              &reference.payload)) {
    if (fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Bypass)
      return std::optional<StorageProjection>{};
    const auto owner = ::loom::fabric::FabricInventoryOwnerRef::of(fifo->owner);
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract || static_cast<std::uint32_t>(
                         ::fabric::FifoResourceState::BufferedQueue) >=
                         contract->stateCount())
      return invalid("selected buffered FIFO has no ResourceContract");
    const auto dimensions = contract->capacityDimensions(
        ::fabric::StateKey(static_cast<std::uint32_t>(
            ::fabric::FifoResourceState::BufferedQueue)));
    const auto queue =
        static_cast<std::uint32_t>(::fabric::FifoBufferedCapacity::QueueSlot);
    if (queue >= dimensions.size() || dimensions[queue].capacity.value() == 0)
      return invalid("selected buffered FIFO has no positive queue capacity");
    const auto patternOwner = ::loom::fabric::FabricUsePatternOwnerRef(owner);
    StorageProjection result;
    result.accessKind = CgraTraversalStorageKind::BufferedFifo;
    result.capacity = dimensions[queue].capacity.value();
    result.key = {0, ::loom::fabric::canonicalFabricBytes(owner), 0};
    result.enqueuePattern = {
        patternOwner,
        ::fabric::fifoUsePattern(::fabric::FifoUsePattern::Enqueue).ordinal()};
    result.dequeuePattern = {
        patternOwner,
        ::fabric::fifoUsePattern(::fabric::FifoUsePattern::Dequeue).ordinal()};
    result.simultaneousPattern = ::loom::fabric::FabricUsePatternRef{
        patternOwner, ::fabric::fifoUsePattern(
                          ::fabric::FifoUsePattern::SimultaneousDequeueEnqueue)
                          .ordinal()};
    return std::optional<StorageProjection>(std::move(result));
  }
  if (const auto *registerFifo =
          std::get_if<::loom::fabric::FabricPeRegisterFifoPayload>(
              &reference.payload)) {
    const auto owner =
        ::loom::fabric::FabricInventoryOwnerRef::of(registerFifo->owner);
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    const std::uint64_t count = fabric.inventorySize(
        owner, ::loom::fabric::FabricInventoryKind::RegisterFifo);
    if (!contract || count > std::numeric_limits<std::uint32_t>::max() ||
        registerFifo->registerFifo >= count)
      return invalid("selected register FIFO has no ResourceContract");
    auto state = ::fabric::resolveTemporalPeRegisterFifoState(
        *contract, static_cast<std::uint32_t>(count),
        static_cast<std::uint32_t>(registerFifo->registerFifo));
    if (!state)
      return state.takeError();
    const auto dimensions = contract->capacityDimensions(*state);
    if (dimensions.empty() || dimensions.front().capacity.value() == 0)
      return invalid("selected register FIFO has no positive queue capacity");
    const auto kind =
        registerFifo->role == ::loom::fabric::FabricRegisterFifoPathRole::Write
            ? CgraTraversalStorageKind::RegisterFifoWrite
            : CgraTraversalStorageKind::RegisterFifoRead;
    auto enqueue = ::fabric::resolveTemporalPeRegisterFifoPattern(
        *contract, static_cast<std::uint32_t>(count),
        static_cast<std::uint32_t>(registerFifo->registerFifo), true);
    if (!enqueue)
      return enqueue.takeError();
    auto dequeue = ::fabric::resolveTemporalPeRegisterFifoPattern(
        *contract, static_cast<std::uint32_t>(count),
        static_cast<std::uint32_t>(registerFifo->registerFifo), false);
    if (!dequeue)
      return dequeue.takeError();
    const auto patternOwner = ::loom::fabric::FabricUsePatternOwnerRef(owner);
    StorageProjection result;
    result.accessKind = kind;
    result.capacity = dimensions.front().capacity.value();
    result.key = {1, ::loom::fabric::canonicalFabricBytes(owner),
                  registerFifo->registerFifo};
    result.enqueuePattern = {patternOwner, enqueue->ordinal()};
    result.dequeuePattern = {patternOwner, dequeue->ordinal()};
    return std::optional<StorageProjection>(std::move(result));
  }
  return std::optional<StorageProjection>{};
}

} // namespace

llvm::Expected<CgraTransportPlan> freezeCgraTransportPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<::dataflow::GraphRef> mappedGraphs,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients) {
  if (physicalUseClients.size() != spatial.resourceUses().size())
    return invalid("CGRA transport physical-use client coverage is incomplete");
  auto operandQueueGroups =
      ::loom::mapping::deriveSpatialPeOperandQueueMatchGroups(
          tech, fabric, spatial.computeBindings(), spatial.routeTrees(),
          spatial.resourceUses(), spatial.physicalTagSegments());
  if (!operandQueueGroups)
    return operandQueueGroups.takeError();
  auto operandQueueProgress =
      ::loom::mapping::deriveSpatialPeOperandProgressFeedback(
          dataflow, tech, *operandQueueGroups);
  if (!operandQueueProgress)
    return operandQueueProgress.takeError();
  auto packedSwitchRows =
      ::loom::mapping::deriveSpatialTemporalSwitchPackedRows(
          fabric, spatial.routeTrees(), spatial.resourceUses(),
          spatial.physicalTagSegments());
  if (!packedSwitchRows)
    return packedSwitchRows.takeError();
  std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef> selected;
  for (const auto &route : spatial.routeTrees()) {
    collect(route.localTraversal, selected);
    for (const auto &node : route.nodes)
      collect(node.incomingTraversal, selected);
    for (const auto &sink : route.sinks)
      collect(sink.localTraversal, selected);
  }
  for (const auto &transfer : spatial.registerFifoTransfers()) {
    selected.try_emplace(bytes(transfer.writeTraversal),
                         transfer.writeTraversal);
    selected.try_emplace(bytes(transfer.readTraversal), transfer.readTraversal);
  }

  std::map<RefBytes, const ::loom::fabric::FabricPhysicalTraversalView *>
      physical;
  for (const auto &traversal : fabric.physicalTraversals())
    if (!physical.try_emplace(bytes(traversal.reference), &traversal).second)
      return invalid("Fabric contains duplicate physical traversal references");

  CgraTransportPlan result;
  result.operandQueueProgress = std::move(*operandQueueProgress);
  std::map<RefBytes, CgraPeOperandBufferPlan> operandBuffers;
  for (const auto &group : *operandQueueGroups)
    for (const auto &match : group.matches) {
      const auto pe = match.queue.context.pe;
      const RefBytes key = ::loom::fabric::canonicalFabricBytes(pe);
      if (operandBuffers.find(key) != operandBuffers.end())
        continue;
      const auto mode = fabric.peOperandBufferMode(pe);
      const std::uint32_t entries = fabric.peOperandBufferSize(pe);
      auto schema = fabric.temporalPeConfigurationSchema(pe);
      if (!mode || entries == 0 || !schema)
        return invalid("CGRA PE operand buffer has no exact Fabric contract");
      CgraPeOperandBufferPlan plan{
          pe, *mode, schema->layout().contextCount, entries, {}};
      plan.fuInputCounts.reserve(schema->layout().fus.size());
      for (const auto &fu : schema->layout().fus)
        plan.fuInputCounts.push_back(fu.inputCount);
      operandBuffers.emplace(key, std::move(plan));
    }
  for (auto &[key, plan] : operandBuffers) {
    (void)key;
    result.operandBuffers.push_back(std::move(plan));
  }
  result.operandQueueActivations.reserve(operandQueueGroups->size());
  for (const auto &group : *operandQueueGroups) {
    auto matchCount =
        checkedU32(group.matches.size(), "CGRA PE operand match count");
    if (!matchCount)
      return matchCount.takeError();
    const std::uint64_t matchOffset = result.operandQueueMatches.size();
    for (const auto &match : group.matches) {
      auto consumerCount =
          checkedU32(match.consumers.size(), "CGRA PE operand consumer count");
      if (!consumerCount)
        return consumerCount.takeError();
      const std::uint64_t consumerOffset = result.operandQueueConsumers.size();
      for (const auto &consumer : match.consumers)
        result.operandQueueConsumers.push_back({consumer});
      result.operandQueueMatches.push_back(
          {match.queue, match.fu, match.allocationUnit, match.entryCapacity,
           consumerOffset, *consumerCount});
    }
    result.operandQueueActivations.push_back(
        {group.logicalNet, group.ingress, group.tag, matchOffset, *matchCount});
  }
  std::vector<std::vector<std::uint64_t>> routeNodeTags;
  std::vector<std::vector<std::uint64_t>> routeNodeSegments;
  routeNodeTags.reserve(spatial.routeTrees().size());
  routeNodeSegments.reserve(spatial.routeTrees().size());
  for (const auto &route : spatial.routeTrees()) {
    routeNodeTags.emplace_back(route.nodes.size(), invalidCgraTransportOrdinal);
    routeNodeSegments.emplace_back(route.nodes.size(),
                                   invalidCgraTransportOrdinal);
  }
  std::vector<std::uint64_t> nextTagSegment(spatial.routeTrees().size(), 0);
  result.physicalTags.reserve(spatial.physicalTagSegments().size());
  for (const auto &segment : spatial.physicalTagSegments()) {
    if (segment.routeTreeOrdinal >= spatial.routeTrees().size() ||
        segment.resourceUseOrdinal >= spatial.resourceUses().size() ||
        segment.segmentOrdinal != nextTagSegment[segment.routeTreeOrdinal]++ ||
        segment.nodeOrdinals.empty())
      return invalid("CGRA Physical Tag segment projection is malformed");
    const auto &use = spatial.resourceUses()[segment.resourceUseOrdinal];
    if (!use.parameters.empty() || use.sharingAssignments.size() != 1)
      return invalid("CGRA Physical Tag ResourceUse has the wrong shape");
    const auto *tag = std::get_if<::fabric::PhysicalTagPatternValue>(
        &use.sharingAssignments.front());
    if (!tag || tag->value.getBitWidth() == 0)
      return invalid("CGRA Physical Tag ResourceUse has no typed value");
    const std::uint64_t tagOrdinal = result.physicalTags.size();
    result.physicalTags.push_back({tag->value});
    for (std::uint64_t node : segment.nodeOrdinals) {
      if (node >= routeNodeTags[segment.routeTreeOrdinal].size() ||
          routeNodeTags[segment.routeTreeOrdinal][node] !=
              invalidCgraTransportOrdinal)
        return invalid("CGRA Physical Tag segment repeats a RouteTree node");
      routeNodeTags[segment.routeTreeOrdinal][node] = tagOrdinal;
      routeNodeSegments[segment.routeTreeOrdinal][node] =
          segment.segmentOrdinal;
    }
  }
  struct SelectedLocalTransfer final {
    const ::loom::mapping::SpatialRegisterFifoTransferView *transfer = nullptr;
    std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
  };
  std::map<EdgeKey, SelectedLocalTransfer> selectedLocalTransfers;
  for (const auto &transfer : spatial.registerFifoTransfers()) {
    auto producerKey = dataflowBytes(dataflow, transfer.logicalNet);
    auto sinkKey = dataflowBytes(dataflow, transfer.sink);
    if (!producerKey)
      return producerKey.takeError();
    if (!sinkKey)
      return sinkKey.takeError();
    const std::uint64_t tagOrdinal = result.physicalTags.size();
    result.physicalTags.push_back({transfer.tag});
    if (!selectedLocalTransfers
             .try_emplace(EdgeKey{std::move(*producerKey), std::move(*sinkKey)},
                          SelectedLocalTransfer{&transfer, tagOrdinal})
             .second)
      return invalid("CGRA register-FIFO transfer edge is duplicated");
  }
  std::set<EdgeKey> selectedMemoryInternalEdges;
  for (const ::loom::mapping::TechMemoryRealizationView &realization :
       tech.memoryRealizations()) {
    for (const ::loom::mapping::TechMemoryInternalEdgeView &edge :
         realization.internalEdges) {
      auto producerKey = dataflowBytes(dataflow, edge.producer);
      auto consumerKey = dataflowBytes(dataflow, edge.consumer);
      if (!producerKey)
        return producerKey.takeError();
      if (!consumerKey)
        return consumerKey.takeError();
      if (!selectedMemoryInternalEdges
               .emplace(std::move(*producerKey), std::move(*consumerKey))
               .second)
        return invalid("CGRA memory internal edge is duplicated");
    }
  }
  struct ProducedUseBuilder final {
    ::dataflow::CanonicalGraphProducerEndpointRef endpoint;
    std::vector<std::uint64_t> actions;
  };
  struct ConsumedUseBuilder final {
    ::dataflow::CanonicalGraphConsumerEndpointRef endpoint;
    std::vector<std::uint64_t> actions;
  };
  std::map<RefBytes, ProducedUseBuilder> producedUses;
  std::map<RefBytes, ConsumedUseBuilder> consumedUses;
  for (auto [actionOrdinal, use] : llvm::enumerate(spatial.resourceUses())) {
    switch (physicalUseClients[actionOrdinal]) {
    case CgraPhysicalUseClientKind::ComputeTransition:
    case CgraPhysicalUseClientKind::MemoryTransition:
      if (!std::holds_alternative<
              ::loom::mapping::SpatialActorTransitionEventRef>(
              use.activation.trigger.event))
        return invalid("CGRA transition action has an endpoint trigger");
      break;
    case CgraPhysicalUseClientKind::ProducedTransport: {
      const auto *endpoint =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
              &use.activation.trigger.event);
      if (!endpoint)
        return invalid("CGRA Produced action has another trigger kind");
      auto key = dataflowBytes(dataflow, *endpoint);
      if (!key)
        return key.takeError();
      auto [position, inserted] =
          producedUses.try_emplace(*key, ProducedUseBuilder{*endpoint, {}});
      (void)inserted;
      position->second.actions.push_back(actionOrdinal);
      break;
    }
    case CgraPhysicalUseClientKind::ConsumedTransport: {
      const auto *endpoint =
          std::get_if<::dataflow::CanonicalGraphConsumerEndpointRef>(
              &use.activation.trigger.event);
      if (!endpoint)
        return invalid("CGRA Consumed action has another trigger kind");
      auto key = dataflowBytes(dataflow, *endpoint);
      if (!key)
        return key.takeError();
      auto [position, inserted] =
          consumedUses.try_emplace(*key, ConsumedUseBuilder{*endpoint, {}});
      (void)inserted;
      position->second.actions.push_back(actionOrdinal);
      break;
    }
    case CgraPhysicalUseClientKind::TraversalTransport:
      return invalid("CGRA derived traversal action appears in Mapping uses");
    }
  }
  std::map<RefBytes, std::uint64_t> selectedOrdinals;
  std::map<StorageKey, std::uint64_t> storageOrdinals;
  std::map<TraversalActivationKey, std::uint64_t> activationInstances;
  std::uint64_t nextActivationInstance = 0;
  result.traversals.reserve(selected.size());
  for (const auto &[key, reference] : selected) {
    auto found = physical.find(key);
    if (found == physical.end())
      return invalid("selected RouteTree traversal is absent from Fabric");
    auto storage = storageContract(fabric, reference);
    if (!storage)
      return storage.takeError();
    CgraTraversalStorageKind storageKind = CgraTraversalStorageKind::None;
    std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
    if (*storage) {
      storageKind = (*storage)->accessKind;
      auto [position, inserted] = storageOrdinals.try_emplace(
          (*storage)->key, result.traversalStorages.size());
      storageOrdinal = position->second;
      if (inserted) {
        result.traversalStorages.push_back(
            {storageKind, (*storage)->capacity, (*storage)->enqueuePattern,
             (*storage)->dequeuePattern, (*storage)->simultaneousPattern});
      } else {
        CgraTraversalStoragePlan &existing =
            result.traversalStorages[storageOrdinal];
        const bool bothRegister =
            existing.kind != CgraTraversalStorageKind::BufferedFifo &&
            storageKind != CgraTraversalStorageKind::BufferedFifo;
        if ((!bothRegister && existing.kind != storageKind) ||
            existing.capacity != (*storage)->capacity ||
            existing.enqueuePattern != (*storage)->enqueuePattern ||
            existing.dequeuePattern != (*storage)->dequeuePattern ||
            existing.simultaneousPattern != (*storage)->simultaneousPattern)
          return invalid("selected traversals disagree on storage contract");
      }
    }
    const std::uint64_t useOffset = result.traversalUses.size();
    if (found->second->impliedUses.size() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("selected traversal implied-use count exceeds u32");
    const auto *switchPayload =
        std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
            &reference.payload);
    const bool temporalSwitch =
        switchPayload && fabric.switchSchedule(switchPayload->owner) ==
                             ::fabric::Schedule::Temporal;
    if (!temporalSwitch)
      for (const auto &use : found->second->impliedUses) {
        auto key = traversalActivationKey(fabric, reference, use);
        if (!key)
          return key.takeError();
        auto [instance, inserted] = activationInstances.try_emplace(
            std::move(*key), nextActivationInstance);
        if (inserted)
          ++nextActivationInstance;
        result.traversalUses.push_back({use.pattern, use.requesterGroup,
                                        instance->second,
                                        invalidCgraTransportOrdinal});
      }
    const std::uint64_t ordinal = result.traversals.size();
    selectedOrdinals.emplace(key, ordinal);
    result.traversals.push_back(
        {reference, reference.kind(), storageKind, storageOrdinal, useOffset,
         temporalSwitch
             ? 0
             : static_cast<std::uint32_t>(found->second->impliedUses.size())});
  }

  using TemporalActivationMemberKey =
      std::tuple<std::uint64_t, std::uint64_t, RefBytes>;
  struct TemporalActivationSlice final {
    std::uint64_t offset = 0;
    std::uint32_t count = 0;
  };
  std::map<TemporalActivationMemberKey, TemporalActivationSlice>
      temporalActivationSlices;
  for (const auto &row : *packedSwitchRows) {
    std::map<
        ::loom::fabric::FabricOrdinal,
        std::vector<
            const ::loom::mapping::SpatialTemporalSwitchRouteSignatureView *>>
        byInput;
    for (const auto &signature : row.signatures)
      byInput[signature.input].push_back(&signature);
    for (const auto &[input, signatures] : byInput) {
      std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef>
          activationTraversals;
      for (const auto *signature : signatures)
        for (const auto &traversal : signature->traversals)
          activationTraversals.try_emplace(bytes(traversal), traversal);
      const std::uint64_t useOffset = result.traversalUses.size();
      const std::uint64_t activationInstance = nextActivationInstance++;
      for (const auto &[traversalKey, traversal] : activationTraversals) {
        auto physicalTraversal = physical.find(traversalKey);
        if (physicalTraversal == physical.end())
          return invalid(
              "Temporal switch activation traversal is absent from Fabric");
        for (const auto &use : physicalTraversal->second->impliedUses) {
          if (use.requesterGroup.kind !=
                  ::loom::fabric::FabricTraversalRequesterGroupKind::
                      SwitchRequester ||
              use.requesterGroup.owner !=
                  ::loom::fabric::FabricInventoryOwnerRef::of(row.occurrence) ||
              use.requesterGroup.ordinal != input)
            return invalid(
                "Temporal switch traversal has the wrong requester group");
          result.traversalUses.push_back({use.pattern, use.requesterGroup,
                                          activationInstance,
                                          invalidCgraTransportOrdinal});
        }
      }
      auto useCount = checkedU32(result.traversalUses.size() - useOffset,
                                 "CGRA switch activation use count");
      if (!useCount)
        return useCount.takeError();
      if (*useCount == 0)
        return invalid("Temporal switch activation has no resource use");
      const TemporalActivationSlice slice{useOffset, *useCount};
      for (const auto *signature : signatures)
        for (const auto &traversal : signature->traversals)
          if (!temporalActivationSlices
                   .try_emplace(
                       TemporalActivationMemberKey{signature->routeTreeOrdinal,
                                                   signature->segmentOrdinal,
                                                   bytes(traversal)},
                       slice)
                   .second)
            return invalid(
                "Temporal switch route member has multiple activations");
    }
  }

  const auto ordinalOf =
      [&](const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
              &reference) -> llvm::Expected<std::uint64_t> {
    if (!reference)
      return invalidCgraTransportOrdinal;
    auto found = selectedOrdinals.find(bytes(*reference));
    if (found == selectedOrdinals.end())
      return invalid("RouteTree traversal is absent from selected catalog");
    return found->second;
  };

  result.routes.reserve(spatial.routeTrees().size());
  std::set<EdgeKey> residualEdges;
  std::set<RefBytes> transferProducers;
  std::set<RefBytes> transferConsumers;
  for (auto [routeOrdinal, route] : llvm::enumerate(spatial.routeTrees())) {
    auto graph = resolveGraphOf(dataflow, route.logicalNet);
    if (!graph)
      return graph.takeError();
    auto sourceTraversal = ordinalOf(route.localTraversal);
    if (!sourceTraversal)
      return sourceTraversal.takeError();
    auto nodeCount = checkedU32(route.nodes.size(), "CGRA route-node count");
    if (!nodeCount)
      return nodeCount.takeError();
    auto sinkCount = checkedU32(route.sinks.size(), "CGRA route-sink count");
    if (!sinkCount)
      return sinkCount.takeError();
    const std::uint64_t nodeOffset = result.routeNodes.size();
    const std::uint64_t sinkOffset = result.routeSinks.size();
    auto producerKey = dataflowBytes(dataflow, route.logicalNet);
    if (!producerKey)
      return producerKey.takeError();
    transferProducers.insert(*producerKey);
    for (auto [nodeOrdinal, node] : llvm::enumerate(route.nodes)) {
      auto traversal = ordinalOf(node.incomingTraversal);
      if (!traversal)
        return traversal.takeError();
      std::uint64_t impliedUseOffset = invalidCgraTransportOrdinal;
      std::uint32_t impliedUseCount = 0;
      if (node.incomingTraversal) {
        const CgraSelectedTraversalPlan &selectedTraversal =
            result.traversals[*traversal];
        impliedUseOffset = selectedTraversal.impliedUseOffset;
        impliedUseCount = selectedTraversal.impliedUseCount;
        const auto *sw =
            std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
                &node.incomingTraversal->payload);
        if (sw &&
            fabric.switchSchedule(sw->owner) == ::fabric::Schedule::Temporal) {
          const std::uint64_t segment =
              routeNodeSegments[routeOrdinal][nodeOrdinal];
          if (segment == invalidCgraTransportOrdinal)
            return invalid(
                "Temporal switch route node has no Physical Tag segment");
          auto activation =
              temporalActivationSlices.find(TemporalActivationMemberKey{
                  routeOrdinal, segment, bytes(*node.incomingTraversal)});
          if (activation == temporalActivationSlices.end())
            return invalid(
                "Temporal switch route node has no packed-row activation");
          impliedUseOffset = activation->second.offset;
          impliedUseCount = activation->second.count;
          temporalActivationSlices.erase(activation);
        }
      }
      std::uint32_t parent = std::numeric_limits<std::uint32_t>::max();
      if (node.parentOrdinal) {
        auto checkedParent =
            checkedU32(*node.parentOrdinal, "CGRA route parent ordinal");
        if (!checkedParent)
          return checkedParent.takeError();
        parent = *checkedParent;
      }
      result.routeNodes.push_back({parent, *traversal,
                                   routeNodeTags[routeOrdinal][nodeOrdinal],
                                   impliedUseOffset, impliedUseCount});
    }
    for (const auto &sink : route.sinks) {
      auto node = checkedU32(sink.nodeOrdinal, "CGRA route sink node ordinal");
      if (!node)
        return node.takeError();
      auto traversal = ordinalOf(sink.localTraversal);
      if (!traversal)
        return traversal.takeError();
      result.routeSinks.push_back({sink.sink, *node, *traversal});
      auto sinkKey = dataflowBytes(dataflow, sink.sink);
      if (!sinkKey)
        return sinkKey.takeError();
      transferConsumers.insert(*sinkKey);
      if (!residualEdges.emplace(*producerKey, std::move(*sinkKey)).second)
        return invalid("selected RouteTrees contain a duplicate residual edge");
    }
    result.routes.push_back({route.logicalNet, *graph, *sourceTraversal,
                             nodeOffset, *nodeCount, sinkOffset, *sinkCount});
  }
  if (!temporalActivationSlices.empty())
    return invalid(
        "Temporal switch packed-row activation has no RouteTree member");

  std::set<std::uint64_t> coveredGraphs;
  for (const auto &graph : mappedGraphs)
    coveredGraphs.insert(graph.entity.value());
  struct LocalTransferBuilder final {
    ::dataflow::CanonicalGraphProducerEndpointRef producer;
    ::dataflow::GraphRef graph;
    std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> sinks;
    std::uint64_t writeTraversalOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t readTraversalOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
  };
  std::map<RefBytes, LocalTransferBuilder> localTransfers;
  std::set<EdgeKey> consumedLocalTransfers;
  std::set<EdgeKey> consumedMemoryInternalEdges;
  if (llvm::Error error = dataflow.forEachGraphEdge(
          [&](const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
              const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer)
              -> llvm::Error {
            auto graph = resolveGraphOf(dataflow, producer);
            if (!graph)
              return graph.takeError();
            if (!coveredGraphs.count(graph->entity.value()))
              return llvm::Error::success();
            auto producerKey = dataflowBytes(dataflow, producer);
            if (!producerKey)
              return producerKey.takeError();
            auto consumerKey = dataflowBytes(dataflow, consumer);
            if (!consumerKey)
              return consumerKey.takeError();
            const EdgeKey edge{*producerKey, *consumerKey};
            if (selectedMemoryInternalEdges.count(edge)) {
              if (residualEdges.count(edge))
                return invalid("CGRA memory internal edge is also routed");
              if (selectedLocalTransfers.count(edge))
                return invalid(
                    "CGRA memory internal edge also selects a register FIFO");
              consumedMemoryInternalEdges.insert(edge);
              return llvm::Error::success();
            }
            if (residualEdges.count(edge))
              return llvm::Error::success();
            transferProducers.insert(*producerKey);
            transferConsumers.insert(*consumerKey);
            auto [position, inserted] = localTransfers.try_emplace(
                *producerKey, LocalTransferBuilder{producer, *graph, {}});
            auto selectedLocal = selectedLocalTransfers.find(edge);
            if (selectedLocal != selectedLocalTransfers.end()) {
              if (!inserted || !position->second.sinks.empty())
                return invalid("CGRA register-FIFO transfer is not an exact "
                               "single-consumer edge");
              auto write =
                  ordinalOf(selectedLocal->second.transfer->writeTraversal);
              auto read =
                  ordinalOf(selectedLocal->second.transfer->readTraversal);
              if (!write)
                return write.takeError();
              if (!read)
                return read.takeError();
              position->second.writeTraversalOrdinal = *write;
              position->second.readTraversalOrdinal = *read;
              position->second.physicalTagOrdinal =
                  selectedLocal->second.physicalTagOrdinal;
              consumedLocalTransfers.insert(edge);
            } else if (!inserted && position->second.writeTraversalOrdinal !=
                                        invalidCgraTransportOrdinal) {
              return invalid("CGRA register-FIFO producer has another local "
                             "consumer");
            }
            position->second.sinks.push_back(consumer);
            return llvm::Error::success();
          }))
    return std::move(error);
  if (consumedLocalTransfers.size() != selectedLocalTransfers.size())
    return invalid("CGRA register-FIFO transfer is absent from Dataflow");
  if (consumedMemoryInternalEdges.size() != selectedMemoryInternalEdges.size())
    return invalid("CGRA memory internal edge is absent from Dataflow");
  result.localTransfers.reserve(localTransfers.size());
  for (auto &[key, transfer] : localTransfers) {
    (void)key;
    auto sinkCount =
        checkedU32(transfer.sinks.size(), "CGRA local-transfer sink count");
    if (!sinkCount)
      return sinkCount.takeError();
    const std::uint64_t sinkOffset = result.localTransferSinks.size();
    for (auto &sink : transfer.sinks)
      result.localTransferSinks.push_back({std::move(sink)});
    result.localTransfers.push_back(
        {transfer.producer, transfer.graph, sinkOffset, *sinkCount,
         transfer.writeTraversalOrdinal, transfer.readTraversalOrdinal,
         transfer.physicalTagOrdinal});
  }

  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors()) {
    if (!coveredGraphs.count(actor.graph.entity.value()))
      continue;
    for (std::uint64_t ordinal = 0; ordinal != actor.op->getNumResults();
         ++ordinal) {
      const ::dataflow::ActorTokenResultRef resultRef{actor.ref, ordinal};
      auto consumers = dataflow.graphConsumers(
          ::dataflow::CanonicalGraphProducerEndpointRef(resultRef));
      if (!consumers)
        return consumers.takeError();
      if (consumers->empty())
        result.discardedResults.push_back(resultRef);
    }
  }

  result.producedUses.reserve(producedUses.size());
  for (auto &[key, use] : producedUses) {
    if (!transferProducers.count(key))
      return invalid("CGRA Produced action has no selected transfer source");
    auto count =
        checkedU32(use.actions.size(), "CGRA Produced physical-use count");
    if (!count)
      return count.takeError();
    const std::uint64_t offset = result.endpointPhysicalUses.size();
    result.endpointPhysicalUses.insert(result.endpointPhysicalUses.end(),
                                       use.actions.begin(), use.actions.end());
    result.producedUses.push_back({use.endpoint, offset, *count});
  }
  result.consumedUses.reserve(consumedUses.size());
  for (auto &[key, use] : consumedUses) {
    if (!transferConsumers.count(key))
      return invalid("CGRA Consumed action has no selected transfer sink");
    auto count =
        checkedU32(use.actions.size(), "CGRA Consumed physical-use count");
    if (!count)
      return count.takeError();
    const std::uint64_t offset = result.endpointPhysicalUses.size();
    result.endpointPhysicalUses.insert(result.endpointPhysicalUses.end(),
                                       use.actions.begin(), use.actions.end());
    result.consumedUses.push_back({use.endpoint, offset, *count});
  }
  return result;
}

} // namespace loom::sim::detail

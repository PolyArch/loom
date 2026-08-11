#include "CGRATransportPlan.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"

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
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<::dataflow::GraphRef> mappedGraphs,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients) {
  if (physicalUseClients.size() != spatial.resourceUses().size())
    return invalid("CGRA transport physical-use client coverage is incomplete");
  std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef> selected;
  for (const auto &route : spatial.routeTrees()) {
    collect(route.localTraversal, selected);
    for (const auto &node : route.nodes)
      collect(node.incomingTraversal, selected);
    for (const auto &sink : route.sinks)
      collect(sink.localTraversal, selected);
  }

  std::map<RefBytes, const ::loom::fabric::FabricPhysicalTraversalView *>
      physical;
  for (const auto &traversal : fabric.physicalTraversals())
    if (!physical.try_emplace(bytes(traversal.reference), &traversal).second)
      return invalid("Fabric contains duplicate physical traversal references");

  CgraTransportPlan result;
  std::vector<std::vector<std::uint64_t>> routeNodeTags;
  routeNodeTags.reserve(spatial.routeTrees().size());
  for (const auto &route : spatial.routeTrees())
    routeNodeTags.emplace_back(route.nodes.size(), invalidCgraTransportOrdinal);
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
    for (const auto &use : found->second->impliedUses)
      result.traversalUses.push_back(
          {use.pattern, use.activationGroup, invalidCgraTransportOrdinal});
    const std::uint64_t ordinal = result.traversals.size();
    selectedOrdinals.emplace(key, ordinal);
    result.traversals.push_back(
        {reference, reference.kind(), storageKind, storageOrdinal, useOffset,
         static_cast<std::uint32_t>(found->second->impliedUses.size())});
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
      std::uint32_t parent = std::numeric_limits<std::uint32_t>::max();
      if (node.parentOrdinal) {
        auto checkedParent =
            checkedU32(*node.parentOrdinal, "CGRA route parent ordinal");
        if (!checkedParent)
          return checkedParent.takeError();
        parent = *checkedParent;
      }
      result.routeNodes.push_back(
          {parent, *traversal, routeNodeTags[routeOrdinal][nodeOrdinal]});
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

  std::set<std::uint64_t> coveredGraphs;
  for (const auto &graph : mappedGraphs)
    coveredGraphs.insert(graph.entity.value());
  struct LocalTransferBuilder final {
    ::dataflow::CanonicalGraphProducerEndpointRef producer;
    ::dataflow::GraphRef graph;
    std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> sinks;
  };
  std::map<RefBytes, LocalTransferBuilder> localTransfers;
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
            if (residualEdges.count({*producerKey, *consumerKey}))
              return llvm::Error::success();
            transferProducers.insert(*producerKey);
            transferConsumers.insert(*consumerKey);
            auto [position, inserted] = localTransfers.try_emplace(
                *producerKey, LocalTransferBuilder{producer, *graph, {}});
            (void)inserted;
            position->second.sinks.push_back(consumer);
            return llvm::Error::success();
          }))
    return std::move(error);
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
        {transfer.producer, transfer.graph, sinkOffset, *sinkCount});
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

#include "CgraTransportRuntime.h"

#include "Fabric/IR/PhysicalTag.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

using RefBytes = std::vector<std::uint8_t>;

enum class TraversalStepKind : std::uint8_t {
  PhysicalAction,
  BufferedStorage,
  RegisterStorageWrite,
  RegisterStorageRead,
};

struct TraversalStepKey final {
  TraversalStepKind kind = TraversalStepKind::PhysicalAction;
  std::uint64_t ordinal = 0;
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;

  bool operator<(const TraversalStepKey &other) const {
    return std::tie(kind, ordinal, physicalTagOrdinal) <
           std::tie(other.kind, other.ordinal, other.physicalTagOrdinal);
  }
  bool operator==(const TraversalStepKey &other) const {
    return kind == other.kind && ordinal == other.ordinal &&
           physicalTagOrdinal == other.physicalTagOrdinal;
  }
};

std::string traversalStepSetText(const std::set<TraversalStepKey> &steps) {
  std::string result = "[";
  bool first = true;
  for (const TraversalStepKey &step : steps) {
    if (!first)
      result += ",";
    first = false;
    result += std::to_string(static_cast<unsigned>(step.kind));
    result += ":";
    result += std::to_string(step.ordinal);
    result += ":";
    result += std::to_string(step.physicalTagOrdinal);
  }
  result += "]";
  return result;
}

std::string traversalTargetSetText(
    const std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef>
        &targets) {
  std::string result = "[";
  bool first = true;
  for (const auto &[key, target] : targets) {
    (void)key;
    if (!first)
      result += ",";
    first = false;
    result += ::loom::fabric::printFabricRef(target);
  }
  result += "]";
  return result;
}

template <typename Ref>
llvm::Expected<RefBytes>
dataflowBytes(const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const Ref &reference) {
  return ::dataflow::encodeDataflowReference(dataflow.identity(), reference);
}

llvm::Expected<mlir::Value>
resolveObservation(const PreparedGraphExecution &execution,
                   const ::dataflow::GraphEgressTokenRef &egress) {
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<mlir::Value> {
        using Endpoint = std::decay_t<decltype(typed)>;
        llvm::ArrayRef<mlir::Value> values;
        if constexpr (std::is_same_v<Endpoint,
                                     ::dataflow::GraphValueOutputTokenRef>)
          values = execution.returnObservation.values;
        else if constexpr (std::is_same_v<
                               Endpoint, ::dataflow::GraphStreamOutputTokenRef>)
          values = execution.returnObservation.streams;
        else
          values = execution.returnObservation.complete;
        if (typed.ordinal >= values.size())
          return invalid("CGRA transport graph egress is out of range");
        return values[typed.ordinal];
      },
      egress);
}

struct SinkBuilder final {
  ::dataflow::CanonicalGraphConsumerEndpointRef endpoint;
  std::set<TraversalStepKey> terminals;
};

struct BindingBuilder final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::vector<SinkBuilder> sinks;
  std::map<TraversalStepKey, std::set<TraversalStepKey>> traversalPredecessors;
  std::set<TraversalStepKey> traversalTerminals;
  std::map<TraversalStepKey,
           std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef>>
      traversalTargets;
  bool discard = false;
};

struct PhysicalUseSlice final {
  std::uint64_t offset = 0;
  std::uint32_t count = 0;
};

struct OperandQueueProjection final {
  ::fabric::LogicalOperandQueueKey queue;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  std::uint32_t allocationUnit = 0;
  std::uint32_t entryCapacity = 0;
  std::uint64_t activationOrdinal = 0;
};

llvm::Expected<unsigned>
ingressArgumentOrdinal(const ::dataflow::GraphIngressTokenRef &ingress,
                       ::dataflow::GraphRef graphRef,
                       ::dataflow::GraphOp graph) {
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<unsigned> {
        using Endpoint = std::decay_t<decltype(typed)>;
        if (typed.graph != graphRef)
          return invalid("CGRA transport ingress belongs to another graph");
        std::uint64_t ordinal = 0;
        if constexpr (std::is_same_v<Endpoint,
                                     ::dataflow::GraphStartTokenRef>) {
          ordinal = 0;
        } else if constexpr (std::is_same_v<
                                 Endpoint,
                                 ::dataflow::GraphValueInputTokenRef>) {
          ordinal = 1 + typed.ordinal;
        } else {
          const auto segments = graph.getInputSegmentSizes();
          if (segments.empty() || segments.front() < 0)
            return invalid("CGRA transport graph input segments are invalid");
          ordinal =
              1 + static_cast<std::uint64_t>(segments.front()) + typed.ordinal;
        }
        if (ordinal > std::numeric_limits<unsigned>::max() ||
            ordinal >= graph.getBody().front().getNumArguments())
          return invalid("CGRA transport ingress argument is out of range");
        return static_cast<unsigned>(ordinal);
      },
      ingress);
}

} // namespace

CgraTransportRuntime::CgraTransportRuntime(
    const CgraFrozenExecutionPlan &plan, SimulatorState &state,
    CgraPhysicalActionRuntime &physical, std::vector<TransferBinding> bindings,
    std::vector<SinkBinding> sinks,
    std::vector<PublicationBinding> publications,
    std::vector<std::uint32_t> publicationSinks,
    std::vector<std::uint64_t> physicalUses,
    std::vector<TraversalNodeBinding> traversalNodes,
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets,
    std::vector<std::uint64_t> traversalSuccessors,
    std::vector<StorageBinding> storages,
    std::vector<OperandBufferBinding> operandBuffers,
    std::vector<OperandQueueUnitBinding> operandQueueUnits,
    std::vector<OperandQueueBinding> operandQueues,
    llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
        actorSourceBindings,
    llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings,
    llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
        actorInputQueueBindings)
    : plan_(&plan), state_(&state), physical_(&physical),
      bindings_(std::move(bindings)), sinks_(std::move(sinks)),
      publications_(std::move(publications)),
      publicationSinks_(std::move(publicationSinks)),
      physicalUses_(std::move(physicalUses)),
      traversalNodes_(std::move(traversalNodes)),
      traversalTargets_(std::move(traversalTargets)),
      traversalSuccessors_(std::move(traversalSuccessors)),
      storages_(std::move(storages)),
      operandBuffers_(std::move(operandBuffers)),
      operandQueueUnits_(std::move(operandQueueUnits)),
      operandQueues_(std::move(operandQueues)),
      actorSourceBindings_(std::move(actorSourceBindings)),
      ingressSourceBindings_(std::move(ingressSourceBindings)),
      actorInputQueueBindings_(std::move(actorInputQueueBindings)),
      blocked_(bindings_.size()),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {
  actorSourceBindingOrdinals_.resize(state.execution->actorPlans.size());
  for (const auto &[key, binding] : actorSourceBindings_) {
    assert(key.first < actorSourceBindingOrdinals_.size() &&
           "CGRA actor source binding must name a prepared actor");
    actorSourceBindingOrdinals_[key.first].push_back(binding);
  }
  for (auto &bindings : actorSourceBindingOrdinals_) {
    llvm::sort(bindings);
    bindings.erase(std::unique(bindings.begin(), bindings.end()),
                   bindings.end());
  }
  // Intern the plan's Physical Tag values so equal values share one virtual
  // channel regardless of which tag segment produced them. Ranks follow the
  // canonical ascending unsigned value, which is the order a hardware arbiter
  // can rotate through.
  tagVirtualChannelKeys_ =
      internPhysicalTagChannelRanks(plan.transport.physicalTags);
  channelArrivalCounts_.assign(state.channelSlots.size(), 0);
  traversalRemainingPredecessors_.resize(traversalNodes_.size());
  traversalNodeStates_.resize(traversalNodes_.size(), TraversalNodeState::Idle);
  traversalNodeTransferSlots_.resize(traversalNodes_.size(),
                                     invalidCgraTransportOrdinal);
  traversalStorageReserved_.resize(traversalNodes_.size(), false);
  for (const TraversalNodeBinding &origin : traversalNodes_) {
    if (origin.kind != TraversalNodeKind::BufferedStorage ||
        origin.storageOrdinal >= storages_.size())
      continue;
    for (std::uint64_t downstream : origin.downstreamStorageNodes) {
      if (downstream >= traversalNodes_.size())
        continue;
      const std::uint64_t targetStorage =
          traversalNodes_[downstream].storageOrdinal;
      if (targetStorage >= storages_.size() ||
          targetStorage == origin.storageOrdinal)
        continue;
      storages_[targetStorage].upstreamStorageOrdinals.push_back(
          origin.storageOrdinal);
    }
  }
  for (StorageBinding &storage : storages_) {
    llvm::sort(storage.upstreamStorageOrdinals);
    storage.upstreamStorageOrdinals.erase(
        std::unique(storage.upstreamStorageOrdinals.begin(),
                    storage.upstreamStorageOrdinals.end()),
        storage.upstreamStorageOrdinals.end());
  }
  storageFrameCommits_.resize(storages_.size());
  touchedStorageFrameCommits_.reserve(storages_.size());
}

const ::loom::mapping::SpatialPeOperandProgressFeedback &
CgraTransportRuntime::operandQueueProgress() const {
  return plan_->transport.operandQueueProgress;
}

llvm::Expected<CgraTransportRuntime> CgraTransportRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
    SimulatorState &state, CgraPhysicalActionRuntime &physical) {
  auto resolvedGraph = dataflow.resolve(graph);
  if (!resolvedGraph)
    return resolvedGraph.takeError();
  auto graphOp = mlir::dyn_cast<::dataflow::GraphOp>(resolvedGraph->op);
  if (!graphOp)
    return invalid("CGRA transport graph reference is not a graph");
  llvm::DenseMap<mlir::Operation *, std::uint64_t> semanticOrdinals;
  semanticOrdinals.reserve(execution.actorPlans.size());
  for (auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    if (!semanticOrdinals.try_emplace(actor.operation, ordinal).second)
      return invalid("prepared graph contains a duplicate actor operation");

  std::map<RefBytes, PhysicalUseSlice> producedUses;
  std::map<RefBytes, PhysicalUseSlice> consumedUses;
  const auto addPhysicalUseSlice =
      [&](const auto &record, CgraPhysicalUseClientKind expected, auto &catalog,
          const auto &endpoint) -> llvm::Error {
    if (record.physicalUseOffset > plan.transport.endpointPhysicalUses.size() ||
        record.physicalUseCount > plan.transport.endpointPhysicalUses.size() -
                                      record.physicalUseOffset)
      return invalid("CGRA endpoint physical-use slice is malformed");
    for (std::uint64_t action :
         llvm::ArrayRef(plan.transport.endpointPhysicalUses)
             .slice(record.physicalUseOffset, record.physicalUseCount)) {
      if (action >= plan.physicalUseClients.size() ||
          plan.physicalUseClients[action] != expected)
        return invalid("CGRA endpoint physical-use client is inconsistent");
    }
    auto key = dataflowBytes(dataflow, endpoint);
    if (!key)
      return key.takeError();
    if (!catalog
             .try_emplace(std::move(*key),
                          PhysicalUseSlice{record.physicalUseOffset,
                                           record.physicalUseCount})
             .second)
      return invalid("CGRA endpoint has duplicate physical-use slices");
    return llvm::Error::success();
  };
  for (const CgraProducedPhysicalUsePlan &use : plan.transport.producedUses)
    if (llvm::Error error = addPhysicalUseSlice(
            use, CgraPhysicalUseClientKind::ProducedTransport, producedUses,
            use.producer))
      return std::move(error);
  for (const CgraConsumedPhysicalUsePlan &use : plan.transport.consumedUses)
    if (llvm::Error error = addPhysicalUseSlice(
            use, CgraPhysicalUseClientKind::ConsumedTransport, consumedUses,
            use.consumer))
      return std::move(error);

  std::map<RefBytes, BindingBuilder> builders;
  for (const ::dataflow::ActorTokenResultRef &result :
       plan.transport.discardedResults) {
    auto actor = dataflow.resolve(result.actor);
    if (!actor)
      return actor.takeError();
    if (actor->graph != graph)
      continue;
    auto key = dataflowBytes(dataflow, result);
    if (!key)
      return key.takeError();
    if (!builders
             .try_emplace(*key, BindingBuilder{result, {}, {}, {}, {}, true})
             .second)
      return invalid("CGRA discarded result is duplicated");
  }
  const auto addSink = [&](const auto &transfer,
                           const auto &sink) -> llvm::Error {
    const ::dataflow::CanonicalGraphProducerEndpointRef producer =
        transfer.producer;
    auto key = dataflowBytes(dataflow, producer);
    if (!key)
      return key.takeError();
    auto [position, inserted] = builders.try_emplace(
        *key, BindingBuilder{producer, {}, {}, {}, {}, false});
    if (!inserted && position->second.discard)
      return invalid("CGRA discarded result has a transfer sink");
    position->second.sinks.push_back({sink, {}});
    return llvm::Error::success();
  };
  const auto advanceTraversal =
      [&](BindingBuilder &builder, std::uint64_t traversal,
          std::uint64_t physicalTagOrdinal, std::uint64_t impliedUseOffset,
          std::uint32_t impliedUseCount,
          const std::set<TraversalStepKey> &predecessors)
      -> llvm::Expected<std::set<TraversalStepKey>> {
    if (physicalTagOrdinal != invalidCgraTransportOrdinal &&
        physicalTagOrdinal >= plan.transport.physicalTags.size())
      return invalid("CGRA traversal names an unknown Physical Tag");
    if (traversal == invalidCgraTransportOrdinal)
      return predecessors;
    if (traversal >= plan.transport.traversals.size())
      return invalid("CGRA route selects an unknown traversal");
    const CgraSelectedTraversalPlan &selected =
        plan.transport.traversals[traversal];
    if (selected.storageKind != CgraTraversalStorageKind::None) {
      if (selected.storageOrdinal >= plan.transport.traversalStorages.size())
        return invalid("CGRA traversal storage ordinal is out of range");
      TraversalStepKind kind = TraversalStepKind::BufferedStorage;
      if (selected.storageKind == CgraTraversalStorageKind::RegisterFifoWrite)
        kind = TraversalStepKind::RegisterStorageWrite;
      else if (selected.storageKind ==
               CgraTraversalStorageKind::RegisterFifoRead)
        kind = TraversalStepKind::RegisterStorageRead;
      const TraversalStepKey step{kind, selected.storageOrdinal,
                                  physicalTagOrdinal};
      auto [storagePosition, storageInserted] =
          builder.traversalPredecessors.try_emplace(step, predecessors);
      if (!storageInserted && storagePosition->second != predecessors)
        return invalid("CGRA route reuses one traversal storage at "
                       "incompatible causal positions");
      builder.traversalTargets[step].try_emplace(
          ::loom::fabric::canonicalFabricBytes(selected.reference),
          selected.reference);
      return std::set<TraversalStepKey>{step};
    }
    const std::uint64_t useOffset =
        impliedUseOffset == invalidCgraTransportOrdinal
            ? selected.impliedUseOffset
            : impliedUseOffset;
    const std::uint32_t useCount =
        impliedUseOffset == invalidCgraTransportOrdinal
            ? selected.impliedUseCount
            : impliedUseCount;
    if (useOffset > plan.transport.traversalUses.size() ||
        useCount > plan.transport.traversalUses.size() - useOffset)
      return invalid("CGRA traversal implied-use slice is malformed");
    std::set<TraversalStepKey> actions;
    for (const CgraTraversalUsePlan &use :
         llvm::ArrayRef(plan.transport.traversalUses)
             .slice(useOffset, useCount)) {
      if (use.physicalUseOrdinal >= plan.physicalUseClients.size() ||
          plan.physicalUseClients[use.physicalUseOrdinal] !=
              CgraPhysicalUseClientKind::TraversalTransport)
        return invalid("CGRA traversal action has an inconsistent client");
      actions.insert({TraversalStepKind::PhysicalAction, use.physicalUseOrdinal,
                      physicalTagOrdinal});
    }
    if (actions.empty())
      return predecessors;
    for (TraversalStepKey action : actions) {
      std::set<TraversalStepKey> causalPredecessors = predecessors;
      causalPredecessors.erase(action);
      auto [position, inserted] =
          builder.traversalPredecessors.try_emplace(action, causalPredecessors);
      if (!inserted && position->second != causalPredecessors)
        return invalid(
            llvm::Twine("CGRA route reuses traversal activation action ") +
            llvm::Twine(action.ordinal) + " tag " +
            llvm::Twine(action.physicalTagOrdinal) + " selected by " +
            traversalTargetSetText(builder.traversalTargets[action]) + " and " +
            ::loom::fabric::printFabricRef(selected.reference) +
            " with predecessors " + traversalStepSetText(position->second) +
            " and " + traversalStepSetText(causalPredecessors));
      builder.traversalTargets[action].try_emplace(
          ::loom::fabric::canonicalFabricBytes(selected.reference),
          selected.reference);
    }
    return actions;
  };

  for (const CgraLocalTransferPlan &transfer : plan.transport.localTransfers) {
    if (transfer.graph != graph)
      continue;
    if (transfer.sinkOffset > plan.transport.localTransferSinks.size() ||
        transfer.sinkCount >
            plan.transport.localTransferSinks.size() - transfer.sinkOffset)
      return invalid("CGRA local-transfer sink slice is malformed");
    for (const auto &sink : llvm::ArrayRef(plan.transport.localTransferSinks)
                                .slice(transfer.sinkOffset, transfer.sinkCount))
      if (llvm::Error error = addSink(transfer, sink.sink))
        return std::move(error);
    const bool hasWrite =
        transfer.writeTraversalOrdinal != invalidCgraTransportOrdinal;
    const bool hasRead =
        transfer.readTraversalOrdinal != invalidCgraTransportOrdinal;
    const bool hasTag =
        transfer.physicalTagOrdinal != invalidCgraTransportOrdinal;
    if (hasWrite != hasRead || hasWrite != hasTag)
      return invalid("CGRA local transfer has an incomplete physical path");
    if (!hasWrite)
      continue;
    if (transfer.sinkCount != 1)
      return invalid("CGRA register-FIFO transfer is not single-consumer");
    auto key = dataflowBytes(dataflow, transfer.producer);
    if (!key)
      return key.takeError();
    auto builder = builders.find(*key);
    if (builder == builders.end())
      return invalid("CGRA register-FIFO transfer has no binding");
    auto writeFrontier = advanceTraversal(
        builder->second, transfer.writeTraversalOrdinal,
        transfer.physicalTagOrdinal, invalidCgraTransportOrdinal, 0,
        std::set<TraversalStepKey>{});
    if (!writeFrontier)
      return writeFrontier.takeError();
    auto readFrontier =
        advanceTraversal(builder->second, transfer.readTraversalOrdinal,
                         transfer.physicalTagOrdinal,
                         invalidCgraTransportOrdinal, 0, *writeFrontier);
    if (!readFrontier)
      return readFrontier.takeError();
    builder->second.traversalTerminals.insert(readFrontier->begin(),
                                              readFrontier->end());
    if (builder->second.sinks.empty())
      return invalid("CGRA register-FIFO transfer lost its sink");
    builder->second.sinks.back().terminals = *readFrontier;
  }
  for (const CgraRoutePlan &route : plan.transport.routes) {
    if (route.graph != graph)
      continue;
    auto key = dataflowBytes(dataflow, route.producer);
    if (!key)
      return key.takeError();
    auto [position, inserted] = builders.try_emplace(
        *key, BindingBuilder{route.producer, {}, {}, {}, {}, false});
    if (!inserted && position->second.discard)
      return invalid("CGRA discarded result has a RouteTree");
    BindingBuilder &builder = position->second;
    if (route.nodeOffset > plan.transport.routeNodes.size() ||
        route.nodeCount > plan.transport.routeNodes.size() - route.nodeOffset ||
        route.sinkOffset > plan.transport.routeSinks.size() ||
        route.sinkCount > plan.transport.routeSinks.size() - route.sinkOffset)
      return invalid("CGRA RouteTree execution slice is malformed");
    const auto routeNodes = llvm::ArrayRef(plan.transport.routeNodes)
                                .slice(route.nodeOffset, route.nodeCount);
    if (routeNodes.empty())
      return invalid("CGRA RouteTree has no root node");
    auto rootFrontier = advanceTraversal(builder, route.localTraversalOrdinal,
                                         routeNodes.front().physicalTagOrdinal,
                                         invalidCgraTransportOrdinal, 0,
                                         std::set<TraversalStepKey>{});
    if (!rootFrontier)
      return rootFrontier.takeError();
    std::vector<std::set<TraversalStepKey>> frontiers(routeNodes.size());
    for (auto [ordinal, node] : llvm::enumerate(routeNodes)) {
      if (ordinal == 0) {
        if (node.parentOrdinal != std::numeric_limits<std::uint32_t>::max() ||
            node.incomingTraversalOrdinal != invalidCgraTransportOrdinal)
          return invalid("CGRA RouteTree root has an incoming edge");
        frontiers.front() = *rootFrontier;
        continue;
      }
      if (node.parentOrdinal >= ordinal ||
          node.incomingTraversalOrdinal == invalidCgraTransportOrdinal)
        return invalid("CGRA RouteTree node is not in canonical preorder");
      auto frontier =
          advanceTraversal(builder, node.incomingTraversalOrdinal,
                           node.physicalTagOrdinal, node.impliedUseOffset,
                           node.impliedUseCount, frontiers[node.parentOrdinal]);
      if (!frontier)
        return frontier.takeError();
      frontiers[ordinal] = std::move(*frontier);
    }
    for (const auto &sink : llvm::ArrayRef(plan.transport.routeSinks)
                                .slice(route.sinkOffset, route.sinkCount)) {
      if (sink.nodeOrdinal >= frontiers.size())
        return invalid("CGRA RouteTree sink names an unknown node");
      auto frontier = advanceTraversal(
          builder, sink.localTraversalOrdinal,
          routeNodes[sink.nodeOrdinal].physicalTagOrdinal,
          invalidCgraTransportOrdinal, 0, frontiers[sink.nodeOrdinal]);
      if (!frontier)
        return frontier.takeError();
      builder.traversalTerminals.insert(frontier->begin(), frontier->end());
      builder.sinks.push_back({sink.sink, std::move(*frontier)});
    }
  }
  for (const CgraMemoryInternalConnectionPlan &connection :
       plan.memory.internalConnections) {
    auto producer = dataflow.resolve(connection.producer.actor);
    if (!producer)
      return producer.takeError();
    auto consumer = dataflow.resolve(connection.consumer.actor);
    if (!consumer)
      return consumer.takeError();
    if (producer->graph != consumer->graph)
      return invalid("CGRA memory internal connection spans graphs");
    if (producer->graph != graph)
      continue;
    if (llvm::Error error = addSink(
            connection,
            ::dataflow::CanonicalGraphConsumerEndpointRef(connection.consumer)))
      return std::move(error);
  }

  std::map<RefBytes, OperandQueueProjection> projectedQueueSinks;
  if (!plan.transport.operandQueueMatches.empty() &&
      (plan.transport.operandQueueProgress.pairingKeyCount == 0 ||
       plan.transport.operandQueueProgress.pairingKeyCount !=
           plan.transport.operandQueueMatches.size()))
    return invalid("CGRA operand queue progress projection is unsupported or "
                   "disagrees with its match inventory");
  for (auto [activationOrdinal, activation] :
       llvm::enumerate(plan.transport.operandQueueActivations)) {
    if (activation.matchOffset > plan.transport.operandQueueMatches.size() ||
        activation.matchCount > plan.transport.operandQueueMatches.size() -
                                    activation.matchOffset ||
        activation.matchCount == 0)
      return invalid("CGRA PE operand activation slice is malformed");
    std::optional<::dataflow::GraphRef> activationGraph;
    if (const auto *result = std::get_if<::dataflow::ActorTokenResultRef>(
            &activation.producer)) {
      auto actor = dataflow.resolve(result->actor);
      if (!actor)
        return actor.takeError();
      activationGraph = actor->graph;
    } else {
      activationGraph = std::visit(
          [](const auto &ingress) { return ingress.graph; },
          std::get<::dataflow::GraphIngressTokenRef>(activation.producer));
    }
    if (*activationGraph != graph)
      continue;
    auto producerKey = dataflowBytes(dataflow, activation.producer);
    if (!producerKey)
      return producerKey.takeError();
    auto builder = builders.find(*producerKey);
    if (builder == builders.end())
      return invalid("CGRA PE operand activation has no transfer binding");
    for (const CgraPeOperandQueueMatchPlan &match :
         llvm::ArrayRef(plan.transport.operandQueueMatches)
             .slice(activation.matchOffset, activation.matchCount)) {
      if (match.consumerOffset > plan.transport.operandQueueConsumers.size() ||
          match.consumerCount > plan.transport.operandQueueConsumers.size() -
                                    match.consumerOffset ||
          match.consumerCount == 0)
        return invalid("CGRA PE operand consumer slice is malformed");
      if (activation.ingress.owner !=
          ::loom::fabric::FabricTransportEndpointOwnerRef::of(
              match.queue.context.pe))
        return invalid("CGRA PE operand activation spans physical PEs");
      if (match.entryCapacity == 0)
        return invalid("CGRA PE operand queue has zero entry capacity");
      for (const CgraPeOperandQueueConsumerPlan &consumer :
           llvm::ArrayRef(plan.transport.operandQueueConsumers)
               .slice(match.consumerOffset, match.consumerCount)) {
        if (llvm::none_of(builder->second.sinks, [&](const auto &sink) {
              return sink.endpoint == consumer.consumer;
            }))
          return invalid("CGRA PE operand activation names another transfer "
                         "consumer");
        auto sinkKey = dataflowBytes(dataflow, consumer.consumer);
        if (!sinkKey)
          return sinkKey.takeError();
        if (!projectedQueueSinks
                 .try_emplace(*sinkKey,
                              OperandQueueProjection{
                                  match.queue, match.fu, match.allocationUnit,
                                  match.entryCapacity, activationOrdinal})
                 .second)
          return invalid("CGRA PE operand consumer belongs to multiple atomic "
                         "activations");
        if (consumedUses.find(*sinkKey) == consumedUses.end())
          return invalid(
              "CGRA PE operand activation has no enqueue ResourceUse");
      }
    }
  }

  std::vector<TransferBinding> bindings;
  std::vector<SinkBinding> sinks;
  std::vector<PublicationBinding> publications;
  std::vector<std::uint32_t> publicationSinks;
  std::vector<std::uint64_t> physicalUses;
  std::vector<TraversalNodeBinding> traversalNodes;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets;
  std::vector<std::uint64_t> traversalSuccessors;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings;
  std::map<std::pair<RefBytes, std::uint32_t>, std::uint64_t>
      operandQueueUnitByKey;
  std::map<RefBytes, std::uint64_t> operandBufferByPe;
  std::vector<OperandBufferBinding> operandBuffers;
  operandBuffers.reserve(plan.transport.operandBuffers.size());
  for (const CgraPeOperandBufferPlan &buffer : plan.transport.operandBuffers) {
    auto contract = ::fabric::TemporalOperandBufferContract::create(
        {buffer.pe, buffer.contextCount, buffer.fuInputCounts, buffer.mode,
         buffer.entriesPerAllocationUnit});
    if (!contract)
      return contract.takeError();
    const RefBytes key = ::loom::fabric::canonicalFabricBytes(buffer.pe);
    if (!operandBufferByPe.emplace(key, operandBuffers.size()).second)
      return invalid("CGRA operand-buffer plan repeats a physical PE");
    const std::size_t queueCount = contract->logicalQueues().size();
    const std::uint32_t unitCount = contract->allocationUnitCount();
    operandBuffers.push_back(
        {buffer.pe, std::move(*contract),
         std::vector<std::uint64_t>(queueCount, invalidCgraTransportOrdinal),
         std::vector<std::uint64_t>(unitCount, invalidCgraTransportOrdinal)});
  }
  std::map<::fabric::LogicalOperandQueueKey, std::uint64_t>
      operandQueueBindingByKey;
  std::vector<OperandQueueUnitBinding> operandQueueUnits;
  std::vector<OperandQueueBinding> operandQueues;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorInputQueueBindings;
  bindings.reserve(builders.size());
  for (auto &[key, builder] : builders) {
    const auto produced = producedUses.find(key);
    const std::uint64_t producedUseOffset = physicalUses.size();
    std::uint32_t producedUseCount = 0;
    if (produced != producedUses.end()) {
      const PhysicalUseSlice slice = produced->second;
      physicalUses.insert(physicalUses.end(),
                          plan.transport.endpointPhysicalUses.begin() +
                              slice.offset,
                          plan.transport.endpointPhysicalUses.begin() +
                              slice.offset + slice.count);
      producedUseCount = slice.count;
    }
    if (builder.traversalPredecessors.size() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA traversal node count exceeds u32");
    if (!builder.traversalPredecessors.empty() &&
        builder.traversalTerminals.empty())
      return invalid("CGRA traversal DAG has no terminal action");
    if (builder.traversalTerminals.size() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA traversal terminal count exceeds u32");

    std::set<TraversalStepKey> terminalClosure = builder.traversalTerminals;
    std::vector<TraversalStepKey> closureWork(
        builder.traversalTerminals.begin(), builder.traversalTerminals.end());
    while (!closureWork.empty()) {
      const TraversalStepKey action = closureWork.back();
      closureWork.pop_back();
      auto found = builder.traversalPredecessors.find(action);
      if (found == builder.traversalPredecessors.end())
        return invalid("CGRA traversal terminal has no selected action");
      for (TraversalStepKey predecessor : found->second)
        if (terminalClosure.insert(predecessor).second)
          closureWork.push_back(predecessor);
    }
    if (terminalClosure.size() != builder.traversalPredecessors.size())
      return invalid("CGRA traversal DAG contains an unused action");
    if (builder.sinks.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA transport sink count exceeds u32");

    std::map<TraversalStepKey, std::set<std::uint32_t>> descendantSinks;
    std::map<TraversalStepKey, std::set<std::uint32_t>> terminalSinks;
    for (auto [sinkOrdinal, sink] : llvm::enumerate(builder.sinks)) {
      for (const TraversalStepKey terminal : sink.terminals)
        terminalSinks[terminal].insert(static_cast<std::uint32_t>(sinkOrdinal));
      std::set<TraversalStepKey> visited;
      std::vector<TraversalStepKey> work(sink.terminals.begin(),
                                         sink.terminals.end());
      while (!work.empty()) {
        const TraversalStepKey action = work.back();
        work.pop_back();
        if (!visited.insert(action).second)
          continue;
        auto predecessors = builder.traversalPredecessors.find(action);
        if (predecessors == builder.traversalPredecessors.end())
          return invalid("CGRA sink terminal has no traversal action");
        descendantSinks[action].insert(static_cast<std::uint32_t>(sinkOrdinal));
        work.insert(work.end(), predecessors->second.begin(),
                    predecessors->second.end());
      }
    }

    const std::uint64_t traversalNodeOffset = traversalNodes.size();
    std::map<TraversalStepKey, std::uint64_t> nodeOrdinalByAction;
    for (const auto &[action, predecessors] : builder.traversalPredecessors) {
      (void)predecessors;
      nodeOrdinalByAction.emplace(action, traversalNodeOffset +
                                              nodeOrdinalByAction.size());
    }
    std::map<TraversalStepKey, std::set<TraversalStepKey>> successors;
    for (const auto &[action, predecessors] : builder.traversalPredecessors) {
      successors.try_emplace(action);
      for (TraversalStepKey predecessor : predecessors) {
        if (builder.traversalPredecessors.find(predecessor) ==
            builder.traversalPredecessors.end())
          return invalid("CGRA traversal predecessor is not selected");
        successors[predecessor].insert(action);
      }
    }
    for (const auto &[action, predecessors] : builder.traversalPredecessors) {
      const auto &actionSuccessors = successors[action];
      if (predecessors.size() > std::numeric_limits<std::uint32_t>::max() ||
          actionSuccessors.size() > std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA traversal DAG degree exceeds u32");
      const std::uint64_t successorOffset = traversalSuccessors.size();
      for (TraversalStepKey successor : actionSuccessors)
        traversalSuccessors.push_back(nodeOrdinalByAction.at(successor));
      TraversalNodeKind kind = TraversalNodeKind::PhysicalAction;
      std::uint64_t physicalUseOrdinal = invalidCgraTransportOrdinal;
      std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
      switch (action.kind) {
      case TraversalStepKind::PhysicalAction:
        physicalUseOrdinal = action.ordinal;
        break;
      case TraversalStepKind::BufferedStorage:
        kind = TraversalNodeKind::BufferedStorage;
        storageOrdinal = action.ordinal;
        break;
      case TraversalStepKind::RegisterStorageWrite:
        kind = TraversalNodeKind::RegisterStorageWrite;
        storageOrdinal = action.ordinal;
        break;
      case TraversalStepKind::RegisterStorageRead:
        kind = TraversalNodeKind::RegisterStorageRead;
        storageOrdinal = action.ordinal;
        break;
      }
      auto targetPosition = builder.traversalTargets.find(action);
      if (targetPosition == builder.traversalTargets.end() ||
          targetPosition->second.empty())
        return invalid("CGRA traversal action has no exact target traversal");
      if (targetPosition->second.size() >
          std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA traversal target count exceeds u32");
      const std::uint64_t targetTraversalOffset = traversalTargets.size();
      for (const auto &[key, traversal] : targetPosition->second) {
        (void)key;
        traversalTargets.push_back(traversal);
      }
      const auto descendants = descendantSinks.find(action);
      if (descendants == descendantSinks.end() || descendants->second.empty())
        return invalid("CGRA traversal action reaches no logical sink");
      const auto terminals = terminalSinks.find(action);
      traversalNodes.push_back(
          {kind,
           physicalUseOrdinal,
           storageOrdinal,
           action.physicalTagOrdinal,
           targetTraversalOffset,
           static_cast<std::uint32_t>(targetPosition->second.size()),
           successorOffset,
           static_cast<std::uint32_t>(actionSuccessors.size()),
           static_cast<std::uint32_t>(predecessors.size()),
           builder.traversalTerminals.count(action) != 0,
           std::vector<std::uint32_t>(descendants->second.begin(),
                                      descendants->second.end()),
           terminals == terminalSinks.end()
               ? std::vector<std::uint32_t>()
               : std::vector<std::uint32_t>(terminals->second.begin(),
                                            terminals->second.end()),
           {},
           {}});
    }
    const std::uint32_t traversalNodeCount =
        static_cast<std::uint32_t>(builder.traversalPredecessors.size());
    const std::uint32_t traversalTerminalCount =
        static_cast<std::uint32_t>(builder.traversalTerminals.size());
    for (std::uint64_t nodeOrdinal = traversalNodeOffset;
         nodeOrdinal != traversalNodeOffset + traversalNodeCount;
         ++nodeOrdinal) {
      TraversalNodeBinding &origin = traversalNodes[nodeOrdinal];
      if (origin.kind != TraversalNodeKind::BufferedStorage)
        continue;
      std::set<std::uint64_t> downstreamStorages;
      std::set<std::uint32_t> unbufferedSinks(origin.terminalSinks.begin(),
                                              origin.terminalSinks.end());
      std::set<std::uint64_t> visited;
      std::vector<std::uint64_t> work(
          traversalSuccessors.begin() + origin.successorOffset,
          traversalSuccessors.begin() + origin.successorOffset +
              origin.successorCount);
      while (!work.empty()) {
        const std::uint64_t current = work.back();
        work.pop_back();
        if (current < traversalNodeOffset ||
            current >= traversalNodeOffset + traversalNodeCount ||
            !visited.insert(current).second)
          continue;
        const TraversalNodeBinding &node = traversalNodes[current];
        if (node.kind != TraversalNodeKind::PhysicalAction) {
          downstreamStorages.insert(current);
          continue;
        }
        unbufferedSinks.insert(node.terminalSinks.begin(),
                               node.terminalSinks.end());
        work.insert(work.end(),
                    traversalSuccessors.begin() + node.successorOffset,
                    traversalSuccessors.begin() + node.successorOffset +
                        node.successorCount);
      }
      if (downstreamStorages.empty() && unbufferedSinks.empty())
        return invalid("CGRA buffered traversal reaches no next boundary");
      origin.downstreamStorageNodes.assign(downstreamStorages.begin(),
                                           downstreamStorages.end());
      origin.unbufferedDescendantSinks.assign(unbufferedSinks.begin(),
                                              unbufferedSinks.end());
    }
    std::set<RefBytes> uniqueSinks;
    const std::uint64_t sinkOffset = sinks.size();
    std::uint64_t consumedPhysicalUseCount = 0;
    for (const SinkBuilder &sink : builder.sinks) {
      if (sink.terminals.size() > std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA sink terminal count exceeds u32");
      auto sinkKey = dataflowBytes(dataflow, sink.endpoint);
      if (!sinkKey)
        return sinkKey.takeError();
      if (!uniqueSinks.insert(*sinkKey).second)
        return invalid("CGRA transport contains a duplicate software sink");
      const auto consumed = consumedUses.find(*sinkKey);
      const std::uint64_t consumedUseOffset = physicalUses.size();
      std::uint32_t consumedUseCount = 0;
      if (consumed != consumedUses.end()) {
        const PhysicalUseSlice slice = consumed->second;
        physicalUses.insert(physicalUses.end(),
                            plan.transport.endpointPhysicalUses.begin() +
                                slice.offset,
                            plan.transport.endpointPhysicalUses.begin() +
                                slice.offset + slice.count);
        consumedUseCount = slice.count;
      }
      if (consumedUseCount >
          std::numeric_limits<std::uint32_t>::max() - consumedPhysicalUseCount)
        return invalid("CGRA consumed physical-use count exceeds u32");
      const std::uint32_t consumedLocalActionOffset =
          static_cast<std::uint32_t>(consumedPhysicalUseCount);
      consumedPhysicalUseCount += consumedUseCount;
      if (const auto *operand =
              std::get_if<::dataflow::ActorTokenOperandRef>(&sink.endpoint)) {
        auto actor = dataflow.resolve(operand->actor);
        if (!actor)
          return actor.takeError();
        if (operand->ordinal >= actor->op->getNumOperands())
          return invalid("CGRA transport actor operand is out of range");
        auto channel = execution.channelOrdinals.find(
            &actor->op->getOpOperand(operand->ordinal));
        if (channel == execution.channelOrdinals.end())
          return invalid("CGRA transport actor operand has no channel slot");
        auto semantic = semanticOrdinals.find(actor->op);
        if (semantic == semanticOrdinals.end())
          return invalid("CGRA PE operand consumer has no semantic actor");
        std::uint64_t operandQueueBinding = invalidCgraTransportOrdinal;
        std::uint64_t operandActivationOrdinal = invalidCgraTransportOrdinal;
        auto queueProjection = projectedQueueSinks.find(*sinkKey);
        if (queueProjection != projectedQueueSinks.end()) {
          const OperandQueueProjection &projected = queueProjection->second;
          operandActivationOrdinal = projected.activationOrdinal;
          const RefBytes peKey =
              ::loom::fabric::canonicalFabricBytes(projected.queue.context.pe);
          const auto unitKey = std::make_pair(peKey, projected.allocationUnit);
          auto [unitPosition, insertedUnit] = operandQueueUnitByKey.try_emplace(
              unitKey, operandQueueUnits.size());
          if (insertedUnit) {
            operandQueueUnits.push_back(
                {projected.queue.context.pe, projected.allocationUnit,
                 projected.entryCapacity, 0, 0, std::nullopt, 0});
          } else {
            const OperandQueueUnitBinding &unit =
                operandQueueUnits[unitPosition->second];
            if (unit.pe != projected.queue.context.pe ||
                unit.allocationUnit != projected.allocationUnit ||
                unit.capacity != projected.entryCapacity)
              return invalid("CGRA PE operand allocation-unit projection is "
                             "inconsistent");
          }
          auto [queuePosition, insertedQueue] =
              operandQueueBindingByKey.try_emplace(projected.queue,
                                                   operandQueues.size());
          operandQueueBinding = queuePosition->second;
          if (insertedQueue) {
            if (state.channelSlots[channel->second].ready.size() >
                std::numeric_limits<std::uint32_t>::max())
              return invalid("CGRA PE operand queue occupancy exceeds u32");
            const std::uint32_t initialOccupancy = static_cast<std::uint32_t>(
                state.channelSlots[channel->second].ready.size());
            OperandQueueUnitBinding &unit =
                operandQueueUnits[unitPosition->second];
            if (unit.occupancy > unit.capacity ||
                initialOccupancy > unit.capacity - unit.occupancy)
              return invalid("CGRA PE operand allocation unit starts overfull");
            unit.occupancy += initialOccupancy;
            const auto bufferPosition = operandBufferByPe.find(peKey);
            if (bufferPosition == operandBufferByPe.end())
              return invalid("CGRA PE operand queue has no Fabric contract");
            const auto &contract =
                operandBuffers[bufferPosition->second].contract;
            const auto contractPosition =
                llvm::find(contract.logicalQueues(), projected.queue);
            if (contractPosition == contract.logicalQueues().end())
              return invalid("CGRA PE operand QueueKey is outside its Fabric "
                             "contract");
            const std::uint32_t contractQueue =
                static_cast<std::uint32_t>(std::distance(
                    contract.logicalQueues().begin(), contractPosition));
            if (contract.allocationUnitOf(contractQueue) !=
                    projected.allocationUnit ||
                contract.entriesPerAllocationUnit().value() !=
                    projected.entryCapacity)
              return invalid("CGRA PE operand queue disagrees with its Fabric "
                             "allocation unit");
            OperandBufferBinding &buffer =
                operandBuffers[bufferPosition->second];
            if (buffer.runtimeQueues[contractQueue] !=
                invalidCgraTransportOrdinal)
              return invalid("CGRA PE operand contract queue is duplicated");
            const std::uint32_t contractUnit =
                contract.allocationUnitOf(contractQueue);
            std::uint64_t &runtimeUnit = buffer.runtimeUnits[contractUnit];
            if (runtimeUnit != invalidCgraTransportOrdinal &&
                runtimeUnit != unitPosition->second)
              return invalid("CGRA PE operand contract unit has two runtime "
                             "bindings");
            runtimeUnit = unitPosition->second;
            buffer.runtimeQueues[contractQueue] = operandQueues.size();
            operandQueues.push_back({projected.queue,
                                     projected.fu,
                                     bufferPosition->second,
                                     contractQueue,
                                     unitPosition->second,
                                     initialOccupancy,
                                     {},
                                     {}});
            operandQueues.back().entries.resize(initialOccupancy);
          } else {
            if (operandQueueBinding >= operandQueues.size())
              return invalid("CGRA PE operand queue index is malformed");
            const OperandQueueBinding &queue =
                operandQueues[operandQueueBinding];
            const auto bufferPosition = operandBufferByPe.find(peKey);
            if (queue.queue != projected.queue || queue.fu != projected.fu ||
                bufferPosition == operandBufferByPe.end() ||
                queue.bufferBinding != bufferPosition->second ||
                queue.unitBinding != unitPosition->second ||
                state.channelSlots[channel->second].ready.size() !=
                    queue.occupancy)
              return invalid(
                  "CGRA PE operand broadcast consumers disagree on state");
          }
          OperandQueueBinding &queue = operandQueues[operandQueueBinding];
          if (llvm::any_of(queue.consumers, [&](const auto &consumer) {
                return consumer.semanticActorOrdinal == semantic->second &&
                       consumer.inputOrdinal == operand->ordinal;
              }))
            return invalid(
                "CGRA PE operand queue repeats a broadcast consumer");
          queue.consumers.push_back({channel->second, semantic->second,
                                     static_cast<unsigned>(operand->ordinal)});
          if (!actorInputQueueBindings
                   .try_emplace({semantic->second, operand->ordinal},
                                operandQueueBinding)
                   .second)
            return invalid("CGRA PE operand queue has a duplicate actor input "
                           "binding");
        }
        sinks.push_back({SinkKind::Channel,
                         channel->second,
                         {},
                         consumedUseOffset,
                         consumedUseCount,
                         consumedLocalActionOffset,
                         operandQueueBinding,
                         operandActivationOrdinal,
                         invalidCgraTransportOrdinal,
                         semantic->second,
                         static_cast<std::uint32_t>(operand->ordinal),
                         static_cast<std::uint32_t>(sink.terminals.size())});
      } else {
        if (projectedQueueSinks.find(*sinkKey) != projectedQueueSinks.end())
          return invalid("CGRA PE operand queue projects onto a graph egress");
        auto observed = resolveObservation(
            execution,
            std::get<::dataflow::GraphEgressTokenRef>(sink.endpoint));
        if (!observed)
          return observed.takeError();
        sinks.push_back(
            {SinkKind::Observation, 0, *observed, consumedUseOffset,
             consumedUseCount, consumedLocalActionOffset,
             invalidCgraTransportOrdinal, invalidCgraTransportOrdinal,
             invalidCgraTransportOrdinal, invalidCgraTransportOrdinal,
             std::numeric_limits<std::uint32_t>::max(),
             static_cast<std::uint32_t>(sink.terminals.size())});
      }
    }
    std::vector<std::vector<std::uint32_t>> publicationGroups;
    std::map<std::uint64_t, std::uint32_t> queuePublicationGroups;
    for (std::uint32_t localSink = 0; localSink != builder.sinks.size();
         ++localSink) {
      SinkBinding &selected = sinks[sinkOffset + localSink];
      if (selected.operandActivationOrdinal == invalidCgraTransportOrdinal) {
        publicationGroups.push_back({localSink});
        continue;
      }
      auto [group, inserted] = queuePublicationGroups.try_emplace(
          selected.operandActivationOrdinal, publicationGroups.size());
      if (inserted)
        publicationGroups.emplace_back();
      publicationGroups[group->second].push_back(localSink);
    }
    if (publicationGroups.empty())
      publicationGroups.emplace_back();
    if (publicationGroups.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA publication group count exceeds u32");
    const std::uint64_t publicationOffset = publications.size();
    for (const std::vector<std::uint32_t> &group : publicationGroups) {
      const std::uint64_t groupSinkOffset = publicationSinks.size();
      std::uint32_t groupUseCount = 0;
      const std::uint64_t publicationBinding = publications.size();
      for (std::uint32_t localSink : group) {
        if (localSink >= builder.sinks.size())
          return invalid("CGRA publication group names an unknown sink");
        SinkBinding &selected = sinks[sinkOffset + localSink];
        if (selected.physicalUseCount >
            std::numeric_limits<std::uint32_t>::max() - groupUseCount)
          return invalid("CGRA publication physical-use count exceeds u32");
        groupUseCount += selected.physicalUseCount;
        selected.publicationBinding = publicationBinding;
        publicationSinks.push_back(localSink);
      }
      publications.push_back({groupSinkOffset,
                              static_cast<std::uint32_t>(group.size()),
                              groupUseCount});
    }
    const std::uint64_t bindingOrdinal = bindings.size();
    std::optional<std::uint64_t> semanticActorOrdinal;
    if (const auto *producer =
            std::get_if<::dataflow::ActorTokenResultRef>(&builder.producer)) {
      auto resolvedActor = dataflow.resolve(producer->actor);
      if (!resolvedActor)
        return resolvedActor.takeError();
      if (resolvedActor->graph != graph)
        return invalid("CGRA transport producer belongs to another graph");
      auto actor = semanticOrdinals.find(resolvedActor->op);
      if (actor == semanticOrdinals.end())
        return invalid("CGRA transport producer has no semantic actor binding");
      semanticActorOrdinal = actor->second;
      if (!actorSourceBindings
               .try_emplace({actor->second, producer->ordinal}, bindingOrdinal)
               .second)
        return invalid("CGRA transport producer has duplicate bindings");
    } else {
      auto argument = ingressArgumentOrdinal(
          std::get<::dataflow::GraphIngressTokenRef>(builder.producer), graph,
          graphOp);
      if (!argument)
        return argument.takeError();
      if (!ingressSourceBindings.try_emplace(*argument, bindingOrdinal).second)
        return invalid("CGRA transport ingress has duplicate bindings");
    }
    bindings.push_back({builder.producer, sinkOffset,
                        static_cast<std::uint32_t>(builder.sinks.size()),
                        producedUseOffset, producedUseCount,
                        traversalNodeOffset, traversalNodeCount,
                        traversalTerminalCount,
                        static_cast<std::uint32_t>(consumedPhysicalUseCount),
                        publicationOffset,
                        static_cast<std::uint32_t>(publicationGroups.size()),
                        semanticActorOrdinal, 0, builder.discard, false});
  }

  std::vector<StorageBinding> storages;
  storages.reserve(plan.transport.traversalStorages.size());
  for (const CgraTraversalStoragePlan &storage :
       plan.transport.traversalStorages) {
    if (storage.kind == CgraTraversalStorageKind::None)
      return invalid("CGRA traversal storage has no owner kind");
    if (storage.enqueuePhysicalUseOrdinal >= plan.physicalUseClients.size() ||
        storage.dequeuePhysicalUseOrdinal >= plan.physicalUseClients.size() ||
        plan.physicalUseClients[storage.enqueuePhysicalUseOrdinal] !=
            CgraPhysicalUseClientKind::TraversalTransport ||
        plan.physicalUseClients[storage.dequeuePhysicalUseOrdinal] !=
            CgraPhysicalUseClientKind::TraversalTransport)
      return invalid("CGRA traversal storage action coverage is incomplete");
    if (storage.kind == CgraTraversalStorageKind::BufferedFifo &&
        (storage.simultaneousPhysicalUseOrdinal >=
             plan.physicalUseClients.size() ||
         plan.physicalUseClients[storage.simultaneousPhysicalUseOrdinal] !=
             CgraPhysicalUseClientKind::TraversalTransport))
      return invalid("CGRA buffered storage simultaneous action is absent");
    const bool tagSelective =
        storage.queueDiscipline ==
        ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
    if (tagSelective &&
        (storage.offerAdvancePhysicalUseOrdinal >=
             plan.physicalUseClients.size() ||
         plan.physicalUseClients[storage.offerAdvancePhysicalUseOrdinal] !=
             CgraPhysicalUseClientKind::TraversalTransport))
      return invalid("CGRA virtual channel storage offer-advance action is "
                     "absent");
    if (!tagSelective &&
        storage.offerAdvancePhysicalUseOrdinal != invalidCgraTransportOrdinal)
      return invalid("CGRA strict storage owns an offer-advance action");
    const bool fullReplacementAllowed =
        storage.kind != CgraTraversalStorageKind::BufferedFifo &&
        storage.independentReadWriteServices;
    auto queue = CgraTransportStorageRuntime::create(
        storage.capacity, fullReplacementAllowed, storage.queueDiscipline);
    if (!queue)
      return queue.takeError();
    StorageBinding binding(std::move(*queue), storage.kind,
                           storage.independentReadWriteServices);
    binding.enqueueAction = storage.enqueuePhysicalUseOrdinal;
    binding.dequeueAction = storage.dequeuePhysicalUseOrdinal;
    binding.simultaneousAction = storage.simultaneousPhysicalUseOrdinal;
    binding.offerAdvanceAction = storage.offerAdvancePhysicalUseOrdinal;
    storages.push_back(std::move(binding));
  }
  return CgraTransportRuntime(
      plan, state, physical, std::move(bindings), std::move(sinks),
      std::move(publications), std::move(publicationSinks),
      std::move(physicalUses), std::move(traversalNodes),
      std::move(traversalTargets), std::move(traversalSuccessors),
      std::move(storages), std::move(operandBuffers),
      std::move(operandQueueUnits), std::move(operandQueues),
      std::move(actorSourceBindings), std::move(ingressSourceBindings),
      std::move(actorInputQueueBindings));
}

} // namespace loom::sim::detail

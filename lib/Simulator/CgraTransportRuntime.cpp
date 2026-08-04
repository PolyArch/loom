#include "CgraTransportRuntime.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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

void selectEarlier(std::optional<SpatialEventCoordinate> candidate,
                   std::optional<SpatialEventCoordinate> &selected) {
  if (candidate &&
      (!selected || compareSpatialEventCoordinates(*candidate, *selected) < 0))
    selected = std::move(candidate);
}

bool isAt(const std::optional<SpatialEventCoordinate> &candidate,
          const SpatialEventCoordinate &coordinate) {
  return candidate &&
         compareSpatialEventCoordinates(*candidate, coordinate) == 0;
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

struct BindingBuilder final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> sinks;
  std::map<TraversalStepKey, std::set<TraversalStepKey>> traversalPredecessors;
  std::set<TraversalStepKey> traversalTerminals;
  std::map<TraversalStepKey,
           std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef>>
      traversalTargets;
};

struct PhysicalUseSlice final {
  std::uint64_t offset = 0;
  std::uint32_t count = 0;
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
    std::vector<SinkBinding> sinks, std::vector<std::uint64_t> physicalUses,
    std::vector<TraversalNodeBinding> traversalNodes,
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets,
    std::vector<std::uint64_t> traversalSuccessors,
    std::vector<StorageBinding> storages,
    llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
        actorSourceBindings,
    llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings)
    : plan_(&plan), state_(&state), physical_(&physical),
      bindings_(std::move(bindings)), sinks_(std::move(sinks)),
      physicalUses_(std::move(physicalUses)),
      traversalNodes_(std::move(traversalNodes)),
      traversalTargets_(std::move(traversalTargets)),
      traversalSuccessors_(std::move(traversalSuccessors)),
      storages_(std::move(storages)),
      actorSourceBindings_(std::move(actorSourceBindings)),
      ingressSourceBindings_(std::move(ingressSourceBindings)),
      blocked_(bindings_.size()),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {
  traversalRemainingPredecessors_.resize(traversalNodes_.size());
  traversalNodeStates_.resize(traversalNodes_.size(), TraversalNodeState::Idle);
  traversalNodeTransferSlots_.resize(traversalNodes_.size(),
                                     invalidCgraTransportOrdinal);
  storageFrameCommits_.resize(storages_.size());
  touchedStorageFrameCommits_.reserve(storages_.size());
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
  const auto addSink = [&](const auto &transfer,
                           const auto &sink) -> llvm::Error {
    auto key = dataflowBytes(dataflow, transfer.producer);
    if (!key)
      return key.takeError();
    auto [position, inserted] = builders.try_emplace(
        *key, BindingBuilder{transfer.producer, {}, {}, {}, {}});
    (void)inserted;
    position->second.sinks.push_back(sink);
    return llvm::Error::success();
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
  }
  for (const CgraRoutePlan &route : plan.transport.routes) {
    if (route.graph != graph)
      continue;
    auto key = dataflowBytes(dataflow, route.producer);
    if (!key)
      return key.takeError();
    auto [position, inserted] = builders.try_emplace(
        *key, BindingBuilder{route.producer, {}, {}, {}, {}});
    (void)inserted;
    BindingBuilder &builder = position->second;
    if (route.nodeOffset > plan.transport.routeNodes.size() ||
        route.nodeCount > plan.transport.routeNodes.size() - route.nodeOffset ||
        route.sinkOffset > plan.transport.routeSinks.size() ||
        route.sinkCount > plan.transport.routeSinks.size() - route.sinkOffset)
      return invalid("CGRA RouteTree execution slice is malformed");
    const auto advanceTraversal =
        [&](std::uint64_t traversal, std::uint64_t physicalTagOrdinal,
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
      if (selected.impliedUseOffset > plan.transport.traversalUses.size() ||
          selected.impliedUseCount >
              plan.transport.traversalUses.size() - selected.impliedUseOffset)
        return invalid("CGRA traversal implied-use slice is malformed");
      std::set<TraversalStepKey> actions;
      for (const CgraTraversalUsePlan &use :
           llvm::ArrayRef(plan.transport.traversalUses)
               .slice(selected.impliedUseOffset, selected.impliedUseCount)) {
        if (use.physicalUseOrdinal >= plan.physicalUseClients.size() ||
            plan.physicalUseClients[use.physicalUseOrdinal] !=
                CgraPhysicalUseClientKind::TraversalTransport)
          return invalid("CGRA traversal action has an inconsistent client");
        actions.insert({TraversalStepKind::PhysicalAction,
                        use.physicalUseOrdinal, physicalTagOrdinal});
      }
      if (actions.empty())
        return predecessors;
      for (TraversalStepKey action : actions) {
        std::set<TraversalStepKey> causalPredecessors = predecessors;
        causalPredecessors.erase(action);
        auto [position, inserted] = builder.traversalPredecessors.try_emplace(
            action, causalPredecessors);
        if (!inserted && position->second != causalPredecessors)
          return invalid("CGRA route reuses one traversal activation at "
                         "incompatible causal positions");
        builder.traversalTargets[action].try_emplace(
            ::loom::fabric::canonicalFabricBytes(selected.reference),
            selected.reference);
      }
      return actions;
    };
    const auto routeNodes = llvm::ArrayRef(plan.transport.routeNodes)
                                .slice(route.nodeOffset, route.nodeCount);
    if (routeNodes.empty())
      return invalid("CGRA RouteTree has no root node");
    auto rootFrontier = advanceTraversal(route.localTraversalOrdinal,
                                         routeNodes.front().physicalTagOrdinal,
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
      auto frontier = advanceTraversal(node.incomingTraversalOrdinal,
                                       node.physicalTagOrdinal,
                                       frontiers[node.parentOrdinal]);
      if (!frontier)
        return frontier.takeError();
      frontiers[ordinal] = std::move(*frontier);
    }
    for (const auto &sink : llvm::ArrayRef(plan.transport.routeSinks)
                                .slice(route.sinkOffset, route.sinkCount)) {
      if (sink.nodeOrdinal >= frontiers.size())
        return invalid("CGRA RouteTree sink names an unknown node");
      auto frontier =
          advanceTraversal(sink.localTraversalOrdinal,
                           routeNodes[sink.nodeOrdinal].physicalTagOrdinal,
                           frontiers[sink.nodeOrdinal]);
      if (!frontier)
        return frontier.takeError();
      builder.traversalTerminals.insert(frontier->begin(), frontier->end());
      builder.sinks.push_back(sink.sink);
    }
  }

  std::vector<TransferBinding> bindings;
  std::vector<SinkBinding> sinks;
  std::vector<std::uint64_t> physicalUses;
  std::vector<TraversalNodeBinding> traversalNodes;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets;
  std::vector<std::uint64_t> traversalSuccessors;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings;
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
      traversalNodes.push_back(
          {kind, physicalUseOrdinal, storageOrdinal, action.physicalTagOrdinal,
           targetTraversalOffset,
           static_cast<std::uint32_t>(targetPosition->second.size()),
           successorOffset, static_cast<std::uint32_t>(actionSuccessors.size()),
           static_cast<std::uint32_t>(predecessors.size()),
           builder.traversalTerminals.count(action) != 0});
    }
    const std::uint32_t traversalNodeCount =
        static_cast<std::uint32_t>(builder.traversalPredecessors.size());
    const std::uint32_t traversalTerminalCount =
        static_cast<std::uint32_t>(builder.traversalTerminals.size());
    std::set<RefBytes> uniqueSinks;
    const std::uint64_t sinkOffset = sinks.size();
    std::uint64_t consumedPhysicalUseCount = 0;
    for (const auto &sink : builder.sinks) {
      auto sinkKey = dataflowBytes(dataflow, sink);
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
      consumedPhysicalUseCount += consumedUseCount;
      if (const auto *operand =
              std::get_if<::dataflow::ActorTokenOperandRef>(&sink)) {
        auto actor = dataflow.resolve(operand->actor);
        if (!actor)
          return actor.takeError();
        if (operand->ordinal >= actor->op->getNumOperands())
          return invalid("CGRA transport actor operand is out of range");
        auto channel = execution.channelOrdinals.find(
            &actor->op->getOpOperand(operand->ordinal));
        if (channel == execution.channelOrdinals.end())
          return invalid("CGRA transport actor operand has no channel slot");
        sinks.push_back({SinkKind::Channel,
                         channel->second,
                         {},
                         consumedUseOffset,
                         consumedUseCount});
      } else {
        auto observed = resolveObservation(
            execution, std::get<::dataflow::GraphEgressTokenRef>(sink));
        if (!observed)
          return observed.takeError();
        sinks.push_back({SinkKind::Observation, 0, *observed, consumedUseOffset,
                         consumedUseCount});
      }
    }
    if (builder.sinks.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA transport sink count exceeds u32");
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
                        semanticActorOrdinal, 0, false});
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
    auto queue = CgraTransportStorageRuntime::create(storage.capacity);
    if (!queue)
      return queue.takeError();
    StorageBinding binding(std::move(*queue), storage.kind,
                           storage.independentReadWriteServices);
    binding.enqueueAction = storage.enqueuePhysicalUseOrdinal;
    binding.dequeueAction = storage.dequeuePhysicalUseOrdinal;
    binding.simultaneousAction = storage.simultaneousPhysicalUseOrdinal;
    storages.push_back(std::move(binding));
  }
  return CgraTransportRuntime(
      plan, state, physical, std::move(bindings), std::move(sinks),
      std::move(physicalUses), std::move(traversalNodes),
      std::move(traversalTargets), std::move(traversalSuccessors),
      std::move(storages), std::move(actorSourceBindings),
      std::move(ingressSourceBindings));
}

llvm::Error CgraTransportRuntime::scheduleArrival(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA transport arrival names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  if (inFlight.arrivalScheduled || inFlight.consumedRequested)
    return invalid("CGRA transport arrival was scheduled twice");
  arrivalEvents_.schedule(
      {{coordinate, inFlight.bindingOrdinal, inFlight.occurrenceOrdinal, 0},
       slot});
  inFlight.arrivalScheduled = true;
  return llvm::Error::success();
}

llvm::Expected<bool> CgraTransportRuntime::scheduleReadyTraversals(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA traversal event names an inactive token");
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  bool scheduled = false;
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    if (traversalNodeStates_[nodeOrdinal] != TraversalNodeState::Idle ||
        traversalRemainingPredecessors_[nodeOrdinal] != 0)
      continue;
    const TraversalNodeBinding &node = traversalNodes_[nodeOrdinal];
    if (node.kind != TraversalNodeKind::PhysicalAction) {
      if (node.storageOrdinal >= storages_.size())
        return invalid("CGRA traversal storage ordinal is out of range");
      StorageBinding &storage = storages_[node.storageOrdinal];
      const bool buffered =
          node.kind == TraversalNodeKind::BufferedStorage &&
          storage.kind == CgraTraversalStorageKind::BufferedFifo;
      const bool registerWrite =
          node.kind == TraversalNodeKind::RegisterStorageWrite &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo;
      const bool registerRead =
          node.kind == TraversalNodeKind::RegisterStorageRead &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo;
      if (!buffered && !registerWrite && !registerRead)
        return invalid("CGRA traversal disagrees with its storage owner");
      if (registerRead)
        storage.pendingDequeueNodes.push_back(nodeOrdinal);
      else
        storage.pendingEnqueueNodes.push_back(nodeOrdinal);
      traversalNodeStates_[nodeOrdinal] = TraversalNodeState::WaitingStorage;
      if (llvm::Error error = scheduleStorage(node.storageOrdinal, coordinate))
        return std::move(error);
      scheduled = true;
      continue;
    }
    traversalEvents_.schedule(
        {{coordinate, nodeOrdinal, inFlight.occurrenceOrdinal,
          static_cast<std::uint32_t>(nodeOrdinal -
                                     binding.traversalNodeOffset)},
         nodeOrdinal});
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Scheduled;
    scheduled = true;
  }
  return scheduled;
}

llvm::Error CgraTransportRuntime::scheduleStorage(
    std::uint64_t storageOrdinal, const SpatialEventCoordinate &coordinate) {
  if (storageOrdinal >= storages_.size())
    return invalid("CGRA storage event names an unknown queue");
  StorageBinding &storage = storages_[storageOrdinal];
  if (storage.eventScheduled || storage.activeActionCount != 0)
    return llvm::Error::success();
  if (storage.pendingEnqueueNodes.empty() &&
      storage.pendingDequeueNodes.empty() && storage.queue.empty())
    return llvm::Error::success();
  storageEvents_.schedule({{coordinate, storageOrdinal, 0, 0}, storageOrdinal});
  storage.eventScheduled = true;
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::schedulePublication(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA transport publication names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  if (inFlight.publicationScheduled || inFlight.published)
    return invalid("CGRA transport publication was scheduled twice");
  scheduleAt(slot, coordinate);
  inFlight.publicationScheduled = true;
  return llvm::Error::success();
}

llvm::Expected<std::vector<CgraTransportCompletion>>
CgraTransportRuntime::acceptPhysicalEvents(
    const CgraPhysicalLifecycleFrame &physicalFrame) {
  struct CountDelta final {
    std::uint32_t producedPermitted = 0;
    std::uint32_t producedRetired = 0;
    std::uint32_t traversalPermitted = 0;
    std::uint32_t traversalRetired = 0;
    std::uint32_t traversalTerminalsPermitted = 0;
    std::uint32_t consumedPermitted = 0;
    std::uint32_t consumedRetired = 0;
  };
  using ActionKey = std::pair<std::uint64_t, std::uint64_t>;
  llvm::DenseMap<ActionKey, ActionLifecycleState> projectedStates;
  llvm::DenseMap<std::uint64_t, CountDelta> countDeltas;
  for (std::uint64_t storageOrdinal : touchedStorageFrameCommits_)
    storageFrameCommits_[storageOrdinal] = StorageFrameCommit{};
  touchedStorageFrameCommits_.clear();
  llvm::SmallVector<std::pair<std::uint64_t, std::uint64_t>, 8>
      newlyPermittedTraversals;
  const auto addTraversalPermission = [&](std::uint64_t slot,
                                          std::uint64_t node) -> llvm::Error {
    CountDelta &delta = countDeltas[slot];
    if (delta.traversalPermitted == std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA storage permit count exceeds u32");
    ++delta.traversalPermitted;
    newlyPermittedTraversals.emplace_back(slot, node);
    if (traversalNodes_[node].terminal) {
      if (delta.traversalTerminalsPermitted ==
          std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA storage terminal count exceeds u32");
      ++delta.traversalTerminalsPermitted;
    }
    return llvm::Error::success();
  };

  for (const CgraPhysicalLifecycleEvent &event : physicalFrame.events) {
    if (compareSpatialEventCoordinates(event.coordinate,
                                       physicalFrame.coordinate) != 0)
      return invalid("CGRA physical frame contains another coordinate");
    if (event.kind == CgraPhysicalLifecycleKind::Requested)
      return invalid("CGRA physical runtime repeated a request event");
    if (event.actionOrdinal >= plan_->physicalUseClients.size() ||
        event.actionOrdinal >= plan_->physicalUseTimings.size())
      return invalid("CGRA physical lifecycle names an unknown action");
    const CgraPhysicalUseClientKind client =
        plan_->physicalUseClients[event.actionOrdinal];
    if (client == CgraPhysicalUseClientKind::ComputeTransition ||
        client == CgraPhysicalUseClientKind::MemoryTransition)
      continue;

    const ActionKey key{event.actionOrdinal, event.occurrenceOrdinal};
    auto indexed = actionOwners_.find(key);
    if (indexed == actionOwners_.end())
      return invalid("CGRA physical lifecycle has no transport owner");
    const ActionOwner &owner = indexed->second;
    if (owner.transferSlot >= inFlight_.size() ||
        !inFlight_[owner.transferSlot].active)
      return invalid("CGRA physical lifecycle names an inactive transfer");
    const bool matchingClient =
        (owner.stage == ActionStage::Produced &&
         client == CgraPhysicalUseClientKind::ProducedTransport) ||
        (owner.stage == ActionStage::Traversal &&
         client == CgraPhysicalUseClientKind::TraversalTransport) ||
        (owner.stage == ActionStage::Storage &&
         client == CgraPhysicalUseClientKind::TraversalTransport) ||
        (owner.stage == ActionStage::Consumed &&
         client == CgraPhysicalUseClientKind::ConsumedTransport);
    if (!matchingClient)
      return invalid("CGRA physical lifecycle disagrees with transport stage");
    if (owner.stage == ActionStage::Traversal) {
      const InFlight &inFlight = inFlight_[owner.transferSlot];
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      if (owner.traversalNodeOrdinal < binding.traversalNodeOffset ||
          owner.traversalNodeOrdinal >=
              binding.traversalNodeOffset + binding.traversalNodeCount ||
          traversalNodeTransferSlots_[owner.traversalNodeOrdinal] !=
              owner.transferSlot)
        return invalid("CGRA traversal lifecycle names another transfer DAG");
    } else if (owner.stage == ActionStage::Storage) {
      if (owner.storageOrdinal >= storages_.size())
        return invalid("CGRA storage lifecycle names an unknown queue");
      const StorageBinding &storage = storages_[owner.storageOrdinal];
      const auto expectedStorageNodeState =
          [&](bool enqueue) -> TraversalNodeState {
        if (owner.state != ActionLifecycleState::Permitted)
          return TraversalNodeState::Requested;
        if (enqueue && storage.kind == CgraTraversalStorageKind::BufferedFifo)
          return TraversalNodeState::Queued;
        return TraversalNodeState::Permitted;
      };
      const bool primaryIsEnqueue =
          owner.storageOperation == StorageOperation::Enqueue;
      if (storage.activeActionCount == 0 ||
          owner.traversalNodeOrdinal >= traversalNodes_.size() ||
          traversalNodeTransferSlots_[owner.traversalNodeOrdinal] !=
              owner.transferSlot ||
          traversalNodeStates_[owner.traversalNodeOrdinal] !=
              expectedStorageNodeState(primaryIsEnqueue))
        return invalid("CGRA storage lifecycle names inconsistent state");
      if (owner.secondaryTraversalNodeOrdinal != invalidCgraTransportOrdinal) {
        if (owner.secondaryTransferSlot >= inFlight_.size() ||
            !inFlight_[owner.secondaryTransferSlot].active ||
            owner.secondaryTraversalNodeOrdinal >= traversalNodes_.size() ||
            traversalNodeTransferSlots_[owner.secondaryTraversalNodeOrdinal] !=
                owner.secondaryTransferSlot ||
            traversalNodeStates_[owner.secondaryTraversalNodeOrdinal] !=
                expectedStorageNodeState(true))
          return invalid(
              "CGRA simultaneous storage lifecycle has inconsistent state");
      }
      const std::uint64_t expectedAction =
          owner.storageOperation == StorageOperation::Enqueue
              ? storage.enqueueAction
          : owner.storageOperation == StorageOperation::Dequeue
              ? storage.dequeueAction
              : storage.simultaneousAction;
      if (owner.storageOperation == StorageOperation::None ||
          expectedAction != event.actionOrdinal)
        return invalid("CGRA storage lifecycle uses the wrong pattern");
    } else if (owner.traversalNodeOrdinal != invalidCgraTransportOrdinal) {
      return invalid("CGRA endpoint lifecycle carries a traversal node");
    }

    auto projected = projectedStates.find(key);
    ActionLifecycleState state =
        projected == projectedStates.end() ? owner.state : projected->second;
    const bool requiresCommit =
        plan_->physicalUseTimings[event.actionOrdinal].commitRank.has_value();
    bool permitted = false;
    bool retired = false;
    switch (event.kind) {
    case CgraPhysicalLifecycleKind::Requested:
      llvm_unreachable("request lifecycle rejected above");
    case CgraPhysicalLifecycleKind::Granted:
      if (state != ActionLifecycleState::Requested)
        return invalid("CGRA transport action was granted twice");
      state = requiresCommit ? ActionLifecycleState::Granted
                             : ActionLifecycleState::Permitted;
      permitted = !requiresCommit;
      break;
    case CgraPhysicalLifecycleKind::Committed:
      if (!requiresCommit || state != ActionLifecycleState::Granted)
        return invalid("CGRA transport action commit is inconsistent");
      state = ActionLifecycleState::Permitted;
      permitted = true;
      break;
    case CgraPhysicalLifecycleKind::Retired:
      if (state != ActionLifecycleState::Permitted)
        return invalid("CGRA transport action retired before permission");
      state = ActionLifecycleState::Retired;
      retired = true;
      break;
    }
    projectedStates[key] = state;
    CountDelta &delta = countDeltas[owner.transferSlot];
    std::uint32_t *permittedDelta = nullptr;
    std::uint32_t *retiredDelta = nullptr;
    switch (owner.stage) {
    case ActionStage::Produced:
      permittedDelta = &delta.producedPermitted;
      retiredDelta = &delta.producedRetired;
      break;
    case ActionStage::Traversal:
      permittedDelta = &delta.traversalPermitted;
      retiredDelta = &delta.traversalRetired;
      if (permitted) {
        if (traversalNodeStates_[owner.traversalNodeOrdinal] !=
            TraversalNodeState::Requested)
          return invalid("CGRA traversal permission preceded its request");
        newlyPermittedTraversals.emplace_back(owner.transferSlot,
                                              owner.traversalNodeOrdinal);
        if (traversalNodes_[owner.traversalNodeOrdinal].terminal) {
          if (delta.traversalTerminalsPermitted ==
              std::numeric_limits<std::uint32_t>::max())
            return invalid("CGRA traversal terminal count exceeds u32");
          ++delta.traversalTerminalsPermitted;
        }
      }
      break;
    case ActionStage::Storage: {
      StorageBinding &storage = storages_[owner.storageOrdinal];
      StorageFrameCommit &commit = storageFrameCommits_[owner.storageOrdinal];
      if (!commit.touched) {
        commit.touched = true;
        touchedStorageFrameCommits_.push_back(owner.storageOrdinal);
      }
      if (retired) {
        if (commit.retireCount == std::numeric_limits<std::uint8_t>::max())
          return invalid("CGRA storage retire count exceeds u8");
        ++commit.retireCount;
      }
      const bool hasDequeue =
          owner.storageOperation == StorageOperation::Dequeue ||
          owner.storageOperation == StorageOperation::Simultaneous;
      const bool hasEnqueue =
          owner.storageOperation == StorageOperation::Enqueue ||
          owner.storageOperation == StorageOperation::Simultaneous;
      const std::uint64_t dequeueSlot = owner.transferSlot;
      const std::uint64_t dequeueNode = owner.traversalNodeOrdinal;
      const std::uint64_t enqueueSlot =
          owner.storageOperation == StorageOperation::Simultaneous
              ? owner.secondaryTransferSlot
              : owner.transferSlot;
      const std::uint64_t enqueueNode =
          owner.storageOperation == StorageOperation::Simultaneous
              ? owner.secondaryTraversalNodeOrdinal
              : owner.traversalNodeOrdinal;
      if (permitted && hasDequeue) {
        if (storage.queue.empty() ||
            storage.queue.front().transferSlot != dequeueSlot)
          return invalid("CGRA storage dequeue changed before commit");
        const CgraTransportStorageEntry head = storage.queue.front();
        if (head.traversalNodeOrdinal >= traversalNodes_.size() ||
            head.physicalTagOrdinal !=
                traversalNodes_[head.traversalNodeOrdinal].physicalTagOrdinal)
          return invalid("CGRA storage queue changed its Physical Tag");
        if (storage.kind == CgraTraversalStorageKind::BufferedFifo &&
            head.traversalNodeOrdinal != dequeueNode)
          return invalid("CGRA buffered storage dequeue changed before commit");
        if (storage.kind != CgraTraversalStorageKind::BufferedFifo &&
            (traversalNodes_[head.traversalNodeOrdinal].kind !=
                 TraversalNodeKind::RegisterStorageWrite ||
             traversalNodes_[dequeueNode].kind !=
                 TraversalNodeKind::RegisterStorageRead ||
             head.physicalTagOrdinal !=
                 traversalNodes_[dequeueNode].physicalTagOrdinal))
          return invalid("CGRA register storage roles are inconsistent");
        if (commit.expectedDequeue)
          return invalid("CGRA storage frame contains two dequeues");
        if (storage.kind != CgraTraversalStorageKind::BufferedFifo &&
            llvm::find(storage.pendingDequeueNodes, dequeueNode) ==
                storage.pendingDequeueNodes.end())
          return invalid("CGRA storage dequeue request is not pending");
        commit.expectedDequeue = head;
        commit.dequeueNode = dequeueNode;
        if (llvm::Error error =
                addTraversalPermission(dequeueSlot, dequeueNode))
          return std::move(error);
      }
      if (retired && hasDequeue) {
        CountDelta &dequeueDelta = countDeltas[dequeueSlot];
        if (dequeueDelta.traversalRetired ==
            std::numeric_limits<std::uint32_t>::max())
          return invalid("CGRA storage retire count exceeds u32");
        ++dequeueDelta.traversalRetired;
      }
      if (hasEnqueue && (enqueueSlot >= inFlight_.size() ||
                         enqueueNode >= traversalNodes_.size()))
        return invalid("CGRA storage enqueue owner is out of range");
      if (permitted && hasEnqueue) {
        const TraversalNodeKind expectedKind =
            storage.kind == CgraTraversalStorageKind::BufferedFifo
                ? TraversalNodeKind::BufferedStorage
                : TraversalNodeKind::RegisterStorageWrite;
        if (traversalNodes_[enqueueNode].kind != expectedKind)
          return invalid("CGRA storage enqueue has the wrong traversal role");
        if (commit.enqueue)
          return invalid("CGRA storage frame contains two enqueues");
        if (llvm::find(storage.pendingEnqueueNodes, enqueueNode) ==
            storage.pendingEnqueueNodes.end())
          return invalid("CGRA storage enqueue request is not pending");
        commit.enqueue = CgraTransportStorageEntry{
            enqueueSlot, enqueueNode,
            traversalNodes_[enqueueNode].physicalTagOrdinal};
        commit.enqueueNode = enqueueNode;
        if (storage.kind != CgraTraversalStorageKind::BufferedFifo)
          if (llvm::Error error =
                  addTraversalPermission(enqueueSlot, enqueueNode))
            return std::move(error);
      }
      if (retired && hasEnqueue &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo) {
        CountDelta &enqueueDelta = countDeltas[enqueueSlot];
        if (enqueueDelta.traversalRetired ==
            std::numeric_limits<std::uint32_t>::max())
          return invalid("CGRA storage retire count exceeds u32");
        ++enqueueDelta.traversalRetired;
      }
      continue;
    }
    case ActionStage::Consumed:
      permittedDelta = &delta.consumedPermitted;
      retiredDelta = &delta.consumedRetired;
      break;
    }
    if (permitted &&
        *permittedDelta == std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA transport permit count exceeds u32");
    if (retired && *retiredDelta == std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA transport retire count exceeds u32");
    *permittedDelta += permitted;
    *retiredDelta += retired;
  }

  for (std::uint64_t storageOrdinal : touchedStorageFrameCommits_) {
    const StorageFrameCommit &commit = storageFrameCommits_[storageOrdinal];
    const StorageBinding &storage = storages_[storageOrdinal];
    if (commit.retireCount > storage.activeActionCount)
      return invalid("CGRA storage frame retires too many actions");
    if ((commit.enqueue || commit.expectedDequeue) &&
        !storage.queue.admits(commit.enqueue.has_value(),
                              commit.expectedDequeue.has_value()))
      return invalid("CGRA storage frame violates cycle-start capacity");
  }

  llvm::DenseMap<std::uint64_t, std::uint32_t> successorDeltas;
  for (const auto &[slot, nodeOrdinal] : newlyPermittedTraversals) {
    const TraversalNodeBinding &node = traversalNodes_[nodeOrdinal];
    if (node.successorOffset > traversalSuccessors_.size() ||
        node.successorCount >
            traversalSuccessors_.size() - node.successorOffset)
      return invalid("CGRA traversal successor slice is malformed");
    for (std::uint64_t successor :
         llvm::ArrayRef(traversalSuccessors_)
             .slice(node.successorOffset, node.successorCount)) {
      if (successor >= traversalNodes_.size() ||
          traversalNodeTransferSlots_[successor] != slot ||
          traversalNodeStates_[successor] != TraversalNodeState::Idle)
        return invalid("CGRA traversal successor has inconsistent state");
      std::uint32_t &delta = successorDeltas[successor];
      if (delta == std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA traversal predecessor count exceeds u32");
      ++delta;
      if (delta > traversalRemainingPredecessors_[successor])
        return invalid("CGRA traversal predecessor count underflows");
    }
  }

  bool needsNextDelta = false;
  std::vector<CgraTransportCompletion> completions;
  for (const auto &[slot, delta] : countDeltas) {
    InFlight &inFlight = inFlight_[slot];
    const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    if (inFlight.producedPermitted > binding.physicalUseCount ||
        inFlight.producedRetired > binding.physicalUseCount ||
        inFlight.traversalPermitted > binding.traversalNodeCount ||
        inFlight.traversalRetired > binding.traversalNodeCount ||
        inFlight.traversalTerminalsPermitted > binding.traversalTerminalCount ||
        inFlight.consumedPermitted > binding.consumedPhysicalUseCount ||
        inFlight.consumedRetired > binding.consumedPhysicalUseCount ||
        delta.producedPermitted >
            binding.physicalUseCount - inFlight.producedPermitted ||
        delta.producedRetired >
            binding.physicalUseCount - inFlight.producedRetired ||
        delta.traversalPermitted >
            binding.traversalNodeCount - inFlight.traversalPermitted ||
        delta.traversalRetired >
            binding.traversalNodeCount - inFlight.traversalRetired ||
        delta.traversalTerminalsPermitted >
            binding.traversalTerminalCount -
                inFlight.traversalTerminalsPermitted ||
        delta.consumedPermitted >
            binding.consumedPhysicalUseCount - inFlight.consumedPermitted ||
        delta.consumedRetired >
            binding.consumedPhysicalUseCount - inFlight.consumedRetired)
      return invalid("CGRA transport lifecycle count exceeds selected uses");
    if (delta.consumedPermitted != 0 && !inFlight.consumedRequested)
      return invalid("CGRA consumed action preceded transfer arrival");
    needsNextDelta |=
        (!inFlight.arrivalScheduled && !inFlight.consumedRequested &&
         delta.producedPermitted != 0 &&
         inFlight.producedPermitted + delta.producedPermitted ==
             binding.physicalUseCount) ||
        delta.traversalPermitted != 0 ||
        (!inFlight.publicationScheduled && !inFlight.published &&
         inFlight.consumedRequested && delta.consumedPermitted != 0 &&
         inFlight.consumedPermitted + delta.consumedPermitted ==
             binding.consumedPhysicalUseCount);
  }
  std::optional<SpatialEventCoordinate> next;
  if (needsNextDelta) {
    auto coordinate = nextSpatialDelta(physicalFrame.coordinate);
    if (!coordinate)
      return coordinate.takeError();
    next = std::move(*coordinate);
  }

  llvm::SmallDenseSet<std::uint64_t, 4> storagesToSchedule;
  for (std::uint64_t storageOrdinal : touchedStorageFrameCommits_) {
    const StorageFrameCommit &commit = storageFrameCommits_[storageOrdinal];
    StorageBinding &storage = storages_[storageOrdinal];
    if (!commit.enqueue && !commit.expectedDequeue)
      continue;
    auto committed = storage.queue.commit(commit.enqueue,
                                          commit.expectedDequeue.has_value());
    if (!committed)
      return committed.takeError();
    if (commit.expectedDequeue &&
        (!committed->dequeued ||
         committed->dequeued->transferSlot !=
             commit.expectedDequeue->transferSlot ||
         committed->dequeued->traversalNodeOrdinal !=
             commit.expectedDequeue->traversalNodeOrdinal))
      return invalid("CGRA storage commit dequeued another token");
    if (commit.enqueue) {
      auto pending =
          llvm::find(storage.pendingEnqueueNodes, commit.enqueueNode);
      if (pending == storage.pendingEnqueueNodes.end())
        return invalid("CGRA storage commit lost its enqueue request");
      storage.pendingEnqueueNodes.erase(pending);
      if (storage.kind == CgraTraversalStorageKind::BufferedFifo)
        traversalNodeStates_[commit.enqueueNode] = TraversalNodeState::Queued;
      else
        ++inFlight_[commit.enqueue->transferSlot].traversalPermitted;
    }
    if (commit.expectedDequeue) {
      if (storage.kind != CgraTraversalStorageKind::BufferedFifo) {
        auto pending =
            llvm::find(storage.pendingDequeueNodes, commit.dequeueNode);
        if (pending == storage.pendingDequeueNodes.end())
          return invalid("CGRA storage commit lost its dequeue request");
        storage.pendingDequeueNodes.erase(pending);
      }
      InFlight &dequeued = inFlight_[commit.expectedDequeue->transferSlot];
      if (dequeued.traversalPermitted ==
          std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA storage traversal permit count exceeds u32");
      ++dequeued.traversalPermitted;
    }
  }

  for (const CgraPhysicalLifecycleEvent &event : physicalFrame.events) {
    if (event.actionOrdinal >= plan_->physicalUseClients.size())
      continue;
    const CgraPhysicalUseClientKind client =
        plan_->physicalUseClients[event.actionOrdinal];
    if (client == CgraPhysicalUseClientKind::ComputeTransition ||
        client == CgraPhysicalUseClientKind::MemoryTransition)
      continue;
    const ActionKey key{event.actionOrdinal, event.occurrenceOrdinal};
    auto indexed = actionOwners_.find(key);
    assert(indexed != actionOwners_.end() &&
           "validated transport action owner disappeared");
    ActionOwner &owner = indexed->second;
    InFlight &inFlight = inFlight_[owner.transferSlot];
    const bool requiresCommit =
        plan_->physicalUseTimings[event.actionOrdinal].commitRank.has_value();
    if (owner.stage == ActionStage::Storage) {
      switch (event.kind) {
      case CgraPhysicalLifecycleKind::Requested:
        llvm_unreachable("request lifecycle rejected above");
      case CgraPhysicalLifecycleKind::Granted:
        owner.state = requiresCommit ? ActionLifecycleState::Granted
                                     : ActionLifecycleState::Permitted;
        break;
      case CgraPhysicalLifecycleKind::Committed:
        owner.state = ActionLifecycleState::Permitted;
        break;
      case CgraPhysicalLifecycleKind::Retired: {
        const bool dequeue =
            owner.storageOperation == StorageOperation::Dequeue ||
            owner.storageOperation == StorageOperation::Simultaneous;
        const bool enqueue =
            owner.storageOperation == StorageOperation::Enqueue ||
            owner.storageOperation == StorageOperation::Simultaneous;
        StorageBinding &storage = storages_[owner.storageOrdinal];
        if (dequeue ||
            (enqueue &&
             storage.kind != CgraTraversalStorageKind::BufferedFifo)) {
          if (inFlight.traversalRetired ==
              std::numeric_limits<std::uint32_t>::max())
            return invalid("CGRA storage traversal retire count exceeds u32");
          ++inFlight.traversalRetired;
        }
        if (storage.activeActionCount == 0)
          return invalid("CGRA storage action count underflows");
        --storage.activeActionCount;
        storagesToSchedule.insert(owner.storageOrdinal);
        actionOwners_.erase(indexed);
        break;
      }
      }
      continue;
    }
    std::uint32_t *permittedCount = nullptr;
    std::uint32_t *retiredCount = nullptr;
    switch (owner.stage) {
    case ActionStage::Produced:
      permittedCount = &inFlight.producedPermitted;
      retiredCount = &inFlight.producedRetired;
      break;
    case ActionStage::Traversal:
      permittedCount = &inFlight.traversalPermitted;
      retiredCount = &inFlight.traversalRetired;
      break;
    case ActionStage::Storage:
      llvm_unreachable("storage lifecycle handled above");
    case ActionStage::Consumed:
      permittedCount = &inFlight.consumedPermitted;
      retiredCount = &inFlight.consumedRetired;
      break;
    }
    switch (event.kind) {
    case CgraPhysicalLifecycleKind::Requested:
      llvm_unreachable("request lifecycle rejected above");
    case CgraPhysicalLifecycleKind::Granted:
      owner.state = requiresCommit ? ActionLifecycleState::Granted
                                   : ActionLifecycleState::Permitted;
      if (!requiresCommit)
        ++*permittedCount;
      break;
    case CgraPhysicalLifecycleKind::Committed:
      owner.state = ActionLifecycleState::Permitted;
      ++*permittedCount;
      break;
    case CgraPhysicalLifecycleKind::Retired:
      ++*retiredCount;
      actionOwners_.erase(indexed);
      break;
    }
  }

  for (const auto &[slot, nodeOrdinal] : newlyPermittedTraversals) {
    InFlight &inFlight = inFlight_[slot];
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Permitted;
    if (traversalNodes_[nodeOrdinal].terminal)
      ++inFlight.traversalTerminalsPermitted;
  }
  for (const auto &[successor, delta] : successorDeltas)
    traversalRemainingPredecessors_[successor] -= delta;

  for (const auto &[slot, delta] : countDeltas) {
    InFlight &inFlight = inFlight_[slot];
    const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    if (!inFlight.arrivalScheduled && !inFlight.consumedRequested &&
        delta.producedPermitted != 0 &&
        inFlight.producedPermitted == binding.physicalUseCount) {
      if (!next)
        return invalid("CGRA traversal request lost its next delta");
      if (binding.traversalNodeCount == 0) {
        if (llvm::Error error = scheduleArrival(slot, *next))
          return error;
      } else {
        auto scheduled = scheduleReadyTraversals(slot, *next);
        if (!scheduled)
          return scheduled.takeError();
        if (!*scheduled)
          return invalid("CGRA traversal DAG has no ready root action");
      }
    }
    if (!inFlight.arrivalScheduled && !inFlight.consumedRequested &&
        delta.traversalPermitted != 0 &&
        inFlight.traversalTerminalsPermitted ==
            binding.traversalTerminalCount) {
      if (!next)
        return invalid("CGRA transport arrival lost its next delta");
      if (llvm::Error error = scheduleArrival(slot, *next))
        return error;
    } else if (delta.traversalPermitted != 0) {
      if (!next)
        return invalid("CGRA traversal successor lost its next delta");
      auto scheduled = scheduleReadyTraversals(slot, *next);
      if (!scheduled)
        return scheduled.takeError();
    }
    if (!inFlight.publicationScheduled && !inFlight.published &&
        inFlight.consumedRequested &&
        inFlight.consumedPermitted == binding.consumedPhysicalUseCount) {
      if (!next)
        return invalid("CGRA transport publication lost its next delta");
      if (llvm::Error error = schedulePublication(slot, *next))
        return error;
    }
    if (auto completion = maybeRelease(slot))
      completions.push_back(*completion);
  }
  if (!storagesToSchedule.empty()) {
    if (!next) {
      auto coordinate = nextSpatialDelta(physicalFrame.coordinate);
      if (!coordinate)
        return coordinate.takeError();
      next = std::move(*coordinate);
    }
    for (std::uint64_t storageOrdinal : storagesToSchedule)
      if (llvm::Error error = scheduleStorage(storageOrdinal, *next))
        return std::move(error);
  }
  llvm::sort(completions, [](const CgraTransportCompletion &lhs,
                             const CgraTransportCompletion &rhs) {
    return std::tie(lhs.semanticActorOrdinal, lhs.occurrenceOrdinal) <
           std::tie(rhs.semanticActorOrdinal, rhs.occurrenceOrdinal);
  });
  return completions;
}

bool CgraTransportRuntime::canPublish(const TransferBinding &binding) const {
  for (const SinkBinding &sink :
       llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount))
    if (sink.kind == SinkKind::Channel &&
        !state_->channelSlots[sink.channel].ready.empty())
      return false;
  return true;
}

void CgraTransportRuntime::publish(std::uint64_t slot,
                                   CgraTransportFrame &frame) {
  InFlight &inFlight = inFlight_[slot];
  TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  for (const SinkBinding &sink :
       llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount)) {
    if (sink.kind == SinkKind::Channel) {
      ChannelSlot &channel = state_->channelSlots[sink.channel];
      channel.ready.push_back(inFlight.token);
      if (channel.ownerActorOrdinal != InvalidActorOrdinal) {
        state_->nextActorCandidates.set(channel.ownerActorOrdinal);
        if (state_->execution->actorPlans[channel.ownerActorOrdinal]
                .isPlainMemory())
          state_->plainMemoryCandidates.set(channel.ownerActorOrdinal);
      }
    } else {
      state_->observedOutputs[sink.observation].push_back(inFlight.token);
    }
  }
  frame.publications.push_back({binding.producer, inFlight.occurrenceOrdinal,
                                inFlight.producerSequenceOrdinal,
                                std::move(inFlight.token)});
  inFlight.published = true;
  if (auto completion = maybeRelease(slot))
    frame.completions.push_back(*completion);
}

std::optional<CgraTransportCompletion>
CgraTransportRuntime::maybeRelease(std::uint64_t slot) {
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (inFlight.published &&
      inFlight.producedRetired == binding.physicalUseCount &&
      inFlight.traversalRetired == binding.traversalNodeCount &&
      inFlight.consumedRetired == binding.consumedPhysicalUseCount)
    return release(slot);
  return std::nullopt;
}

std::optional<CgraTransportCompletion>
CgraTransportRuntime::release(std::uint64_t slot) {
  InFlight &inFlight = inFlight_[slot];
  TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  std::optional<CgraTransportCompletion> completion;
  if (binding.semanticActorOrdinal)
    completion = CgraTransportCompletion{*binding.semanticActorOrdinal,
                                         inFlight.occurrenceOrdinal};
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    traversalRemainingPredecessors_[nodeOrdinal] = 0;
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Idle;
    traversalNodeTransferSlots_[nodeOrdinal] = invalidCgraTransportOrdinal;
  }
  binding.active = false;
  blocked_.reset(inFlight.bindingOrdinal);
  inFlight.active = false;
  assert(activeTransferCount_ != 0 &&
         "active CGRA transfer count must not underflow");
  --activeTransferCount_;
  freeSlots_.push_back(slot);
  return completion;
}

llvm::Expected<std::optional<CgraTransportFrame>>
CgraTransportRuntime::advance() {
  const std::optional<SpatialEventCoordinate> coordinate = nextCoordinate();
  if (!coordinate)
    return std::optional<CgraTransportFrame>{};

  CgraTransportFrame frame{*coordinate, {}, {}, {}, {}};
  if (isAt(requestedEvents_.nextCoordinate(), *coordinate)) {
    auto requested = requestedEvents_.popNextFrame();
    if (!requested)
      return requested.takeError();
    for (const CgraScheduledEvent &event : (**requested).events)
      frame.physicalEvents.push_back(
          {CgraPhysicalLifecycleKind::Requested,
           event.order.structuralActionOrdinal, event.order.occurrenceOrdinal,
           event.order.ownerEventOrdinal, event.order.coordinate});
  }

  if (isAt(traversalEvents_.nextCoordinate(), *coordinate)) {
    auto traversals = traversalEvents_.popNextFrame();
    if (!traversals)
      return traversals.takeError();
    llvm::SmallVector<PendingActionTransfer, 4> transfers;
    transfers.reserve((**traversals).events.size());
    for (const CgraScheduledEvent &event : (**traversals).events) {
      const std::uint64_t nodeOrdinal = event.payload;
      if (nodeOrdinal >= traversalNodes_.size() ||
          traversalNodeStates_[nodeOrdinal] != TraversalNodeState::Scheduled)
        return invalid("CGRA traversal event names an unscheduled action");
      const std::uint64_t slot = traversalNodeTransferSlots_[nodeOrdinal];
      if (slot >= inFlight_.size() || !inFlight_[slot].active)
        return invalid("CGRA traversal event names an inactive token");
      InFlight &inFlight = inFlight_[slot];
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      if (nodeOrdinal < binding.traversalNodeOffset ||
          nodeOrdinal >=
              binding.traversalNodeOffset + binding.traversalNodeCount ||
          nodeOrdinal != event.order.structuralActionOrdinal ||
          inFlight.occurrenceOrdinal != event.order.occurrenceOrdinal ||
          event.order.ownerEventOrdinal !=
              nodeOrdinal - binding.traversalNodeOffset)
        return invalid("CGRA traversal event key is inconsistent");
      transfers.push_back({slot, inFlight.bindingOrdinal, nodeOrdinal});
    }
    auto requested = requestActions(transfers, ActionStage::Traversal,
                                    (**traversals).coordinate);
    if (!requested)
      return requested.takeError();
    frame.physicalEvents.insert(frame.physicalEvents.end(), requested->begin(),
                                requested->end());
    for (const PendingActionTransfer &transfer : transfers)
      traversalNodeStates_[transfer.traversalNodeOrdinal] =
          TraversalNodeState::Requested;
  }

  if (isAt(storageEvents_.nextCoordinate(), *coordinate)) {
    auto storageFrame = storageEvents_.popNextFrame();
    if (!storageFrame)
      return storageFrame.takeError();
    for (const CgraScheduledEvent &event : (**storageFrame).events) {
      const std::uint64_t storageOrdinal = event.payload;
      if (storageOrdinal >= storages_.size() ||
          event.order.structuralActionOrdinal != storageOrdinal)
        return invalid("CGRA storage event key is inconsistent");
      StorageBinding &storage = storages_[storageOrdinal];
      if (!storage.eventScheduled || storage.activeActionCount != 0)
        return invalid("CGRA storage event has inconsistent queue state");
      storage.eventScheduled = false;

      const auto pendingLess = [&](std::uint64_t lhs, std::uint64_t rhs) {
        const std::uint64_t lhsSlot = traversalNodeTransferSlots_[lhs];
        const std::uint64_t rhsSlot = traversalNodeTransferSlots_[rhs];
        const InFlight &lhsTransfer = inFlight_[lhsSlot];
        const InFlight &rhsTransfer = inFlight_[rhsSlot];
        return std::tie(lhsTransfer.occurrenceOrdinal,
                        lhsTransfer.bindingOrdinal,
                        lhs) < std::tie(rhsTransfer.occurrenceOrdinal,
                                        rhsTransfer.bindingOrdinal, rhs);
      };
      llvm::sort(storage.pendingEnqueueNodes, pendingLess);
      llvm::sort(storage.pendingDequeueNodes, pendingLess);
      std::optional<std::uint64_t> enqueueNode;
      if (!storage.pendingEnqueueNodes.empty()) {
        const std::uint64_t candidate = storage.pendingEnqueueNodes.front();
        const TraversalNodeKind expectedKind =
            storage.kind == CgraTraversalStorageKind::BufferedFifo
                ? TraversalNodeKind::BufferedStorage
                : TraversalNodeKind::RegisterStorageWrite;
        if (candidate >= traversalNodes_.size() ||
            traversalNodes_[candidate].storageOrdinal != storageOrdinal ||
            traversalNodes_[candidate].kind != expectedKind ||
            traversalNodeStates_[candidate] !=
                TraversalNodeState::WaitingStorage)
          return invalid("CGRA storage enqueue candidate is inconsistent");
        enqueueNode = candidate;
      }
      for (std::uint64_t candidate : storage.pendingDequeueNodes)
        if (candidate >= traversalNodes_.size() ||
            traversalNodes_[candidate].storageOrdinal != storageOrdinal ||
            traversalNodes_[candidate].kind !=
                TraversalNodeKind::RegisterStorageRead ||
            traversalNodeStates_[candidate] !=
                TraversalNodeState::WaitingStorage)
          return invalid("CGRA storage dequeue candidate is inconsistent");

      std::optional<CgraTransportStorageEntry> dequeueEntry;
      std::optional<std::uint64_t> dequeueNode;
      if (!storage.queue.empty()) {
        const CgraTransportStorageEntry &head = storage.queue.front();
        const TraversalNodeKind expectedHeadKind =
            storage.kind == CgraTraversalStorageKind::BufferedFifo
                ? TraversalNodeKind::BufferedStorage
                : TraversalNodeKind::RegisterStorageWrite;
        const TraversalNodeState expectedHeadState =
            storage.kind == CgraTraversalStorageKind::BufferedFifo
                ? TraversalNodeState::Queued
                : TraversalNodeState::Permitted;
        if (head.transferSlot >= inFlight_.size() ||
            !inFlight_[head.transferSlot].active ||
            head.traversalNodeOrdinal >= traversalNodes_.size() ||
            traversalNodes_[head.traversalNodeOrdinal].kind !=
                expectedHeadKind ||
            head.physicalTagOrdinal !=
                traversalNodes_[head.traversalNodeOrdinal].physicalTagOrdinal ||
            traversalNodeStates_[head.traversalNodeOrdinal] !=
                expectedHeadState)
          return invalid("CGRA storage queue head is inconsistent");
        const TransferBinding &binding =
            bindings_[inFlight_[head.transferSlot].bindingOrdinal];
        if (storage.kind == CgraTraversalStorageKind::BufferedFifo &&
            canPublish(binding)) {
          dequeueEntry = head;
          dequeueNode = head.traversalNodeOrdinal;
        } else if (storage.kind != CgraTraversalStorageKind::BufferedFifo &&
                   canPublish(binding)) {
          auto matchingRead = llvm::find_if(
              storage.pendingDequeueNodes, [&](std::uint64_t candidate) {
                return traversalNodeTransferSlots_[candidate] ==
                       head.transferSlot;
              });
          if (matchingRead != storage.pendingDequeueNodes.end()) {
            dequeueEntry = head;
            dequeueNode = *matchingRead;
          }
        }
      }

      const bool dequeue = dequeueEntry.has_value();
      bool enqueue = enqueueNode.has_value() && !storage.queue.full();
      if (enqueueNode && dequeue) {
        if (storage.kind == CgraTraversalStorageKind::BufferedFifo ||
            storage.independentReadWriteServices)
          enqueue = true;
        else
          enqueue = false;
      }
      if (!dequeue && !enqueue) {
        for (std::uint64_t node : storage.pendingEnqueueNodes) {
          const std::uint64_t slot = traversalNodeTransferSlots_[node];
          blocked_.set(inFlight_[slot].bindingOrdinal);
          frame.blockedTransfers.push_back(inFlight_[slot].bindingOrdinal);
        }
        for (std::uint64_t node : storage.pendingDequeueNodes) {
          const std::uint64_t slot = traversalNodeTransferSlots_[node];
          blocked_.set(inFlight_[slot].bindingOrdinal);
          frame.blockedTransfers.push_back(inFlight_[slot].bindingOrdinal);
        }
        if (!storage.queue.empty()) {
          const auto head = storage.queue.front();
          blocked_.set(inFlight_[head.transferSlot].bindingOrdinal);
          frame.blockedTransfers.push_back(
              inFlight_[head.transferSlot].bindingOrdinal);
        }
        continue;
      }

      llvm::SmallVector<CgraPhysicalActionRequest, 2> requests;
      llvm::SmallVector<ActionOwner, 2> owners;
      const auto appendRequest = [&](std::uint64_t action,
                                     ActionOwner owner) -> llvm::Error {
        if (action >= nextActionOccurrence_.size())
          return invalid("CGRA storage operation has no physical action");
        if (nextActionOccurrence_[action] ==
            std::numeric_limits<std::uint64_t>::max())
          return llvm::createStringError(
              std::errc::value_too_large,
              "CGRA storage action occurrence ordinal overflows u64");
        requests.push_back({action, nextActionOccurrence_[action]});
        owners.push_back(std::move(owner));
        return llvm::Error::success();
      };
      const auto assignStorageLocalAction =
          [&](ActionOwner &owner) -> llvm::Error {
        if (owner.transferSlot >= inFlight_.size() ||
            !inFlight_[owner.transferSlot].active)
          return invalid("CGRA storage trace owner is not an active token");
        const TransferBinding &binding =
            bindings_[inFlight_[owner.transferSlot].bindingOrdinal];
        if (owner.traversalNodeOrdinal < binding.traversalNodeOffset ||
            owner.traversalNodeOrdinal >=
                binding.traversalNodeOffset + binding.traversalNodeCount)
          return invalid("CGRA storage trace action names another transfer");
        owner.localActionOrdinal = binding.physicalUseCount +
                                   owner.traversalNodeOrdinal -
                                   binding.traversalNodeOffset;
        return llvm::Error::success();
      };
      if (storage.kind == CgraTraversalStorageKind::BufferedFifo) {
        ActionOwner owner;
        owner.stage = ActionStage::Storage;
        owner.storageOperation = dequeue && enqueue
                                     ? StorageOperation::Simultaneous
                                 : dequeue ? StorageOperation::Dequeue
                                           : StorageOperation::Enqueue;
        owner.storageOrdinal = storageOrdinal;
        owner.state = ActionLifecycleState::Requested;
        if (dequeueEntry) {
          owner.transferSlot = dequeueEntry->transferSlot;
          owner.traversalNodeOrdinal = *dequeueNode;
        }
        if (enqueue) {
          const std::uint64_t enqueueSlot =
              traversalNodeTransferSlots_[*enqueueNode];
          if (dequeueEntry) {
            owner.secondaryTransferSlot = enqueueSlot;
            owner.secondaryTraversalNodeOrdinal = *enqueueNode;
          } else {
            owner.transferSlot = enqueueSlot;
            owner.traversalNodeOrdinal = *enqueueNode;
          }
        }
        const std::uint64_t action =
            owner.storageOperation == StorageOperation::Simultaneous
                ? storage.simultaneousAction
            : owner.storageOperation == StorageOperation::Dequeue
                ? storage.dequeueAction
                : storage.enqueueAction;
        if (llvm::Error error = assignStorageLocalAction(owner))
          return std::move(error);
        if (llvm::Error error = appendRequest(action, std::move(owner)))
          return std::move(error);
      } else {
        if (dequeueEntry) {
          ActionOwner owner;
          owner.transferSlot = dequeueEntry->transferSlot;
          owner.traversalNodeOrdinal = *dequeueNode;
          owner.storageOrdinal = storageOrdinal;
          owner.stage = ActionStage::Storage;
          owner.storageOperation = StorageOperation::Dequeue;
          if (llvm::Error error = assignStorageLocalAction(owner))
            return std::move(error);
          if (llvm::Error error =
                  appendRequest(storage.dequeueAction, std::move(owner)))
            return std::move(error);
        }
        if (enqueue) {
          ActionOwner owner;
          owner.transferSlot = traversalNodeTransferSlots_[*enqueueNode];
          owner.traversalNodeOrdinal = *enqueueNode;
          owner.storageOrdinal = storageOrdinal;
          owner.stage = ActionStage::Storage;
          owner.storageOperation = StorageOperation::Enqueue;
          if (llvm::Error error = assignStorageLocalAction(owner))
            return std::move(error);
          if (llvm::Error error =
                  appendRequest(storage.enqueueAction, std::move(owner)))
            return std::move(error);
        }
      }
      if (owners.size() >
          std::numeric_limits<std::uint8_t>::max() - storage.activeActionCount)
        return invalid("CGRA storage action count exceeds u8");
      llvm::SmallDenseSet<std::pair<std::uint64_t, std::uint64_t>, 2>
          requestKeys;
      for (const CgraPhysicalActionRequest &request : requests) {
        const auto key =
            std::make_pair(request.actionOrdinal, request.occurrenceOrdinal);
        if (actionOwners_.contains(key) || !requestKeys.insert(key).second)
          return invalid("CGRA storage action occurrence is duplicated");
      }
      auto requested = physical_->requestBatch(requests, *coordinate);
      if (!requested)
        return requested.takeError();
      for (auto [request, owner] : llvm::zip(requests, owners)) {
        if (!actionOwners_
                 .try_emplace(std::make_pair(request.actionOrdinal,
                                             request.occurrenceOrdinal),
                              owner)
                 .second)
          return invalid("CGRA storage action occurrence is duplicated");
        ++nextActionOccurrence_[request.actionOrdinal];
        traversalNodeStates_[owner.traversalNodeOrdinal] =
            TraversalNodeState::Requested;
        if (owner.secondaryTraversalNodeOrdinal != invalidCgraTransportOrdinal)
          traversalNodeStates_[owner.secondaryTraversalNodeOrdinal] =
              TraversalNodeState::Requested;
      }
      storage.activeActionCount += static_cast<std::uint8_t>(owners.size());
      frame.physicalEvents.insert(frame.physicalEvents.end(),
                                  requested->begin(), requested->end());
    }
  }

  if (isAt(arrivalEvents_.nextCoordinate(), *coordinate)) {
    auto arrivals = arrivalEvents_.popNextFrame();
    if (!arrivals)
      return arrivals.takeError();
    llvm::SmallVector<PendingActionTransfer, 4> transfers;
    llvm::SmallVector<std::uint64_t, 4> slots;
    transfers.reserve((**arrivals).events.size());
    slots.reserve((**arrivals).events.size());
    for (const CgraScheduledEvent &event : (**arrivals).events) {
      if (event.payload >= inFlight_.size() || !inFlight_[event.payload].active)
        return invalid("CGRA transport arrival names an inactive token");
      InFlight &inFlight = inFlight_[event.payload];
      if (!inFlight.arrivalScheduled || inFlight.consumedRequested ||
          inFlight.bindingOrdinal != event.order.structuralActionOrdinal ||
          inFlight.occurrenceOrdinal != event.order.occurrenceOrdinal)
        return invalid("CGRA transport arrival key is inconsistent");
      transfers.push_back({event.payload, inFlight.bindingOrdinal});
      slots.push_back(event.payload);
    }
    auto requested = requestActions(transfers, ActionStage::Consumed,
                                    (**arrivals).coordinate);
    if (!requested)
      return requested.takeError();
    frame.physicalEvents.insert(frame.physicalEvents.end(), requested->begin(),
                                requested->end());
    for (std::uint64_t slot : slots) {
      InFlight &inFlight = inFlight_[slot];
      inFlight.arrivalScheduled = false;
      inFlight.consumedRequested = true;
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      if (binding.consumedPhysicalUseCount == 0)
        if (llvm::Error error = schedulePublication(slot, *coordinate))
          return error;
    }
  }

  if (isAt(events_.nextCoordinate(), *coordinate)) {
    auto publications = events_.popNextFrame();
    if (!publications)
      return publications.takeError();
    for (const CgraScheduledEvent &event : (**publications).events) {
      if (event.payload >= inFlight_.size() || !inFlight_[event.payload].active)
        return invalid("CGRA transport event names an inactive token");
      InFlight &inFlight = inFlight_[event.payload];
      if (!inFlight.publicationScheduled ||
          inFlight.bindingOrdinal != event.order.structuralActionOrdinal ||
          inFlight.occurrenceOrdinal != event.order.occurrenceOrdinal)
        return invalid("CGRA transport event key disagrees with its token");
      inFlight.publicationScheduled = false;
      TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      if (!canPublish(binding)) {
        blocked_.set(inFlight.bindingOrdinal);
        frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
        continue;
      }
      publish(event.payload, frame);
    }
  }
  llvm::sort(frame.physicalEvents, [](const CgraPhysicalLifecycleEvent &lhs,
                                      const CgraPhysicalLifecycleEvent &rhs) {
    return std::tie(lhs.actionOrdinal, lhs.occurrenceOrdinal,
                    lhs.ownerEventOrdinal, lhs.kind) <
           std::tie(rhs.actionOrdinal, rhs.occurrenceOrdinal,
                    rhs.ownerEventOrdinal, rhs.kind);
  });
  return std::optional<CgraTransportFrame>(std::move(frame));
}

std::optional<SpatialEventCoordinate>
CgraTransportRuntime::nextCoordinate() const {
  std::optional<SpatialEventCoordinate> coordinate;
  selectEarlier(requestedEvents_.nextCoordinate(), coordinate);
  selectEarlier(traversalEvents_.nextCoordinate(), coordinate);
  selectEarlier(storageEvents_.nextCoordinate(), coordinate);
  selectEarlier(arrivalEvents_.nextCoordinate(), coordinate);
  selectEarlier(events_.nextCoordinate(), coordinate);
  return coordinate;
}

llvm::Error
CgraTransportRuntime::retryBlocked(const SpatialEventCoordinate &coordinate) {
  auto publication = nextSpatialDelta(coordinate);
  if (!publication)
    return publication.takeError();
  for (int binding = blocked_.find_first(); binding >= 0;
       binding = blocked_.find_next(binding)) {
    std::optional<std::uint64_t> slot;
    for (auto &&[ordinal, inFlight] : llvm::enumerate(inFlight_))
      if (inFlight.active &&
          inFlight.bindingOrdinal == static_cast<std::uint64_t>(binding)) {
        slot = ordinal;
        break;
      }
    if (!slot)
      return invalid("CGRA blocked transfer has no in-flight token");
    blocked_.reset(binding);
    if (inFlight_[*slot].consumedRequested)
      if (llvm::Error error = schedulePublication(*slot, *publication))
        return error;
  }
  for (std::uint64_t storageOrdinal = 0; storageOrdinal != storages_.size();
       ++storageOrdinal)
    if (llvm::Error error = scheduleStorage(storageOrdinal, *publication))
      return error;
  return llvm::Error::success();
}

} // namespace loom::sim::detail

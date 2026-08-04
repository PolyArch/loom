#include "CgraTransportRuntime.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"

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
  std::map<std::uint64_t, std::set<std::uint64_t>> traversalPredecessors;
  std::set<std::uint64_t> traversalTerminals;
  bool requiresStorageTransport = false;
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
    std::vector<std::uint64_t> traversalSuccessors,
    llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
        actorSourceBindings,
    llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings)
    : plan_(&plan), state_(&state), physical_(&physical),
      bindings_(std::move(bindings)), sinks_(std::move(sinks)),
      physicalUses_(std::move(physicalUses)),
      traversalNodes_(std::move(traversalNodes)),
      traversalSuccessors_(std::move(traversalSuccessors)),
      actorSourceBindings_(std::move(actorSourceBindings)),
      ingressSourceBindings_(std::move(ingressSourceBindings)),
      blocked_(bindings_.size()),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {
  traversalRemainingPredecessors_.resize(traversalNodes_.size());
  traversalNodeStates_.resize(traversalNodes_.size(), TraversalNodeState::Idle);
  traversalNodeTransferSlots_.resize(traversalNodes_.size(),
                                     invalidCgraTransportOrdinal);
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
        *key, BindingBuilder{transfer.producer, {}, {}, {}, false});
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
        *key, BindingBuilder{route.producer, {}, {}, {}, false});
    (void)inserted;
    BindingBuilder &builder = position->second;
    if (route.nodeOffset > plan.transport.routeNodes.size() ||
        route.nodeCount > plan.transport.routeNodes.size() - route.nodeOffset ||
        route.sinkOffset > plan.transport.routeSinks.size() ||
        route.sinkCount > plan.transport.routeSinks.size() - route.sinkOffset)
      return invalid("CGRA RouteTree execution slice is malformed");
    const auto advanceTraversal =
        [&](std::uint64_t traversal,
            const std::set<std::uint64_t> &predecessors)
        -> llvm::Expected<std::set<std::uint64_t>> {
      if (traversal == invalidCgraTransportOrdinal)
        return predecessors;
      if (traversal >= plan.transport.traversals.size())
        return invalid("CGRA route selects an unknown traversal");
      const CgraSelectedTraversalPlan &selected =
          plan.transport.traversals[traversal];
      builder.requiresStorageTransport |=
          selected.storageKind != CgraTraversalStorageKind::None;
      if (selected.impliedUseOffset > plan.transport.traversalUses.size() ||
          selected.impliedUseCount >
              plan.transport.traversalUses.size() - selected.impliedUseOffset)
        return invalid("CGRA traversal implied-use slice is malformed");
      std::set<std::uint64_t> actions;
      for (const CgraTraversalUsePlan &use :
           llvm::ArrayRef(plan.transport.traversalUses)
               .slice(selected.impliedUseOffset, selected.impliedUseCount)) {
        if (use.physicalUseOrdinal >= plan.physicalUseClients.size() ||
            plan.physicalUseClients[use.physicalUseOrdinal] !=
                CgraPhysicalUseClientKind::TraversalTransport)
          return invalid("CGRA traversal action has an inconsistent client");
        actions.insert(use.physicalUseOrdinal);
      }
      if (actions.empty())
        return predecessors;
      for (std::uint64_t action : actions) {
        std::set<std::uint64_t> causalPredecessors = predecessors;
        causalPredecessors.erase(action);
        auto [position, inserted] = builder.traversalPredecessors.try_emplace(
            action, causalPredecessors);
        if (!inserted && position->second != causalPredecessors)
          return invalid("CGRA route reuses one traversal activation at "
                         "incompatible causal positions");
      }
      return actions;
    };
    auto rootFrontier = advanceTraversal(route.localTraversalOrdinal,
                                         std::set<std::uint64_t>{});
    if (!rootFrontier)
      return rootFrontier.takeError();
    const auto routeNodes = llvm::ArrayRef(plan.transport.routeNodes)
                                .slice(route.nodeOffset, route.nodeCount);
    if (routeNodes.empty())
      return invalid("CGRA RouteTree has no root node");
    std::vector<std::set<std::uint64_t>> frontiers(routeNodes.size());
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
                                       frontiers[node.parentOrdinal]);
      if (!frontier)
        return frontier.takeError();
      frontiers[ordinal] = std::move(*frontier);
    }
    for (const auto &sink : llvm::ArrayRef(plan.transport.routeSinks)
                                .slice(route.sinkOffset, route.sinkCount)) {
      if (sink.nodeOrdinal >= frontiers.size())
        return invalid("CGRA RouteTree sink names an unknown node");
      auto frontier = advanceTraversal(sink.localTraversalOrdinal,
                                       frontiers[sink.nodeOrdinal]);
      if (!frontier)
        return frontier.takeError();
      builder.traversalTerminals.insert(frontier->begin(), frontier->end());
      builder.sinks.push_back(sink.sink);
    }
  }

  std::map<std::uint64_t, std::uint64_t> actorPlanByEntity;
  for (auto [ordinal, actor] : llvm::enumerate(plan.computeActors))
    if (actor.graph == graph &&
        !actorPlanByEntity.emplace(actor.actor.entity.value(), ordinal).second)
      return invalid("CGRA transport found duplicate compute actor bindings");

  std::vector<TransferBinding> bindings;
  std::vector<SinkBinding> sinks;
  std::vector<std::uint64_t> physicalUses;
  std::vector<TraversalNodeBinding> traversalNodes;
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

    std::set<std::uint64_t> terminalClosure = builder.traversalTerminals;
    std::vector<std::uint64_t> closureWork(builder.traversalTerminals.begin(),
                                           builder.traversalTerminals.end());
    while (!closureWork.empty()) {
      const std::uint64_t action = closureWork.back();
      closureWork.pop_back();
      auto found = builder.traversalPredecessors.find(action);
      if (found == builder.traversalPredecessors.end())
        return invalid("CGRA traversal terminal has no selected action");
      for (std::uint64_t predecessor : found->second)
        if (terminalClosure.insert(predecessor).second)
          closureWork.push_back(predecessor);
    }
    if (terminalClosure.size() != builder.traversalPredecessors.size())
      return invalid("CGRA traversal DAG contains an unused action");

    const std::uint64_t traversalNodeOffset = traversalNodes.size();
    std::map<std::uint64_t, std::uint64_t> nodeOrdinalByAction;
    for (const auto &[action, predecessors] : builder.traversalPredecessors) {
      (void)predecessors;
      nodeOrdinalByAction.emplace(action, traversalNodeOffset +
                                              nodeOrdinalByAction.size());
    }
    std::map<std::uint64_t, std::set<std::uint64_t>> successors;
    for (const auto &[action, predecessors] : builder.traversalPredecessors) {
      successors.try_emplace(action);
      for (std::uint64_t predecessor : predecessors) {
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
      for (std::uint64_t successor : actionSuccessors)
        traversalSuccessors.push_back(nodeOrdinalByAction.at(successor));
      traversalNodes.push_back(
          {action, successorOffset,
           static_cast<std::uint32_t>(actionSuccessors.size()),
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
    std::optional<std::uint64_t> actorPlanOrdinal;
    if (const auto *producer =
            std::get_if<::dataflow::ActorTokenResultRef>(&builder.producer)) {
      auto actor = actorPlanByEntity.find(producer->actor.entity.value());
      if (actor == actorPlanByEntity.end())
        return invalid("CGRA transport producer has no compute actor binding");
      actorPlanOrdinal = actor->second;
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
    bindings.push_back(
        {builder.producer, sinkOffset,
         static_cast<std::uint32_t>(builder.sinks.size()), producedUseOffset,
         producedUseCount, traversalNodeOffset, traversalNodeCount,
         traversalTerminalCount,
         static_cast<std::uint32_t>(consumedPhysicalUseCount), actorPlanOrdinal,
         builder.requiresStorageTransport, false});
  }
  return CgraTransportRuntime(
      plan, state, physical, std::move(bindings), std::move(sinks),
      std::move(physicalUses), std::move(traversalNodes),
      std::move(traversalSuccessors), std::move(actorSourceBindings),
      std::move(ingressSourceBindings));
}

std::uint64_t CgraTransportRuntime::allocate(std::uint64_t bindingOrdinal,
                                             std::uint64_t occurrenceOrdinal,
                                             Token token) {
  assert(bindingOrdinal < bindings_.size() &&
         !bindings_[bindingOrdinal].active &&
         "CGRA transport allocation requires a validated source");
  assert(activeTransferCount_ != std::numeric_limits<std::uint64_t>::max() &&
         "preflighted active transfer count must fit u64");
  std::uint64_t slot = 0;
  if (freeSlots_.empty()) {
    slot = inFlight_.size();
    inFlight_.emplace_back();
  } else {
    slot = freeSlots_.back();
    freeSlots_.pop_back();
  }
  inFlight_[slot] =
      InFlight{true, bindingOrdinal, occurrenceOrdinal, std::move(token)};
  TransferBinding &binding = bindings_[bindingOrdinal];
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    traversalRemainingPredecessors_[nodeOrdinal] =
        traversalNodes_[nodeOrdinal].predecessorCount;
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Idle;
    traversalNodeTransferSlots_[nodeOrdinal] = slot;
  }
  binding.active = true;
  ++activeTransferCount_;
  return slot;
}

llvm::Expected<std::vector<CgraPhysicalLifecycleEvent>>
CgraTransportRuntime::requestActions(
    llvm::ArrayRef<PendingActionTransfer> transfers, ActionStage stage,
    const SpatialEventCoordinate &coordinate) {
  llvm::SmallVector<CgraPhysicalActionRequest, 8> requests;
  llvm::SmallVector<ActionOwner, 8> owners;
  llvm::DenseMap<std::uint64_t, std::uint64_t> increments;

  CgraPhysicalUseClientKind expectedClient =
      CgraPhysicalUseClientKind::ProducedTransport;
  switch (stage) {
  case ActionStage::Produced:
    break;
  case ActionStage::Traversal:
    expectedClient = CgraPhysicalUseClientKind::TraversalTransport;
    break;
  case ActionStage::Consumed:
    expectedClient = CgraPhysicalUseClientKind::ConsumedTransport;
    break;
  }
  const auto appendAction = [&](std::uint64_t transferSlot,
                                std::uint64_t traversalNodeOrdinal,
                                std::uint64_t action) -> llvm::Error {
    if (action >= nextActionOccurrence_.size() ||
        action >= plan_->physicalUseClients.size() ||
        plan_->physicalUseClients[action] != expectedClient)
      return invalid("CGRA transport action has an inconsistent client");
    const std::uint64_t increment = increments.lookup(action);
    const std::uint64_t next = nextActionOccurrence_[action];
    if (increment == std::numeric_limits<std::uint64_t>::max() ||
        next >= std::numeric_limits<std::uint64_t>::max() - increment)
      return llvm::createStringError(
          std::errc::value_too_large,
          "CGRA transport action occurrence ordinal overflows u64");
    requests.push_back({action, next + increment});
    owners.push_back({transferSlot, traversalNodeOrdinal, stage,
                      ActionLifecycleState::Requested});
    increments[action] = increment + 1;
    return llvm::Error::success();
  };

  for (const PendingActionTransfer &transfer : transfers) {
    if (transfer.bindingOrdinal >= bindings_.size())
      return invalid("CGRA transport action names an unknown binding");
    const TransferBinding &binding = bindings_[transfer.bindingOrdinal];
    if (stage == ActionStage::Produced) {
      for (std::uint64_t action :
           llvm::ArrayRef(physicalUses_)
               .slice(binding.physicalUseOffset, binding.physicalUseCount))
        if (llvm::Error error = appendAction(
                transfer.transferSlot, invalidCgraTransportOrdinal, action))
          return error;
      continue;
    }
    if (stage == ActionStage::Traversal) {
      if (transfer.traversalNodeOrdinal < binding.traversalNodeOffset ||
          transfer.traversalNodeOrdinal >=
              binding.traversalNodeOffset + binding.traversalNodeCount)
        return invalid("CGRA traversal action names another transfer DAG");
      const std::uint64_t action =
          traversalNodes_[transfer.traversalNodeOrdinal].physicalUseOrdinal;
      if (llvm::Error error = appendAction(
              transfer.transferSlot, transfer.traversalNodeOrdinal, action))
        return error;
      continue;
    }
    for (const SinkBinding &sink :
         llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount))
      for (std::uint64_t action :
           llvm::ArrayRef(physicalUses_)
               .slice(sink.physicalUseOffset, sink.physicalUseCount))
        if (llvm::Error error = appendAction(
                transfer.transferSlot, invalidCgraTransportOrdinal, action))
          return error;
  }

  if (requests.empty())
    return std::vector<CgraPhysicalLifecycleEvent>{};
  auto requested = physical_->requestBatch(requests, coordinate);
  if (!requested)
    return requested.takeError();

  for (const auto &[action, increment] : increments)
    nextActionOccurrence_[action] += increment;
  for (auto [request, owner] : llvm::zip(requests, owners)) {
    const bool inserted =
        actionOwners_
            .try_emplace(std::make_pair(request.actionOrdinal,
                                        request.occurrenceOrdinal),
                         owner)
            .second;
    assert(inserted && "preflighted transport action must be unique");
  }
  return requested;
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
    traversalEvents_.schedule(
        {{coordinate, node.physicalUseOrdinal, inFlight.occurrenceOrdinal,
          static_cast<std::uint32_t>(nodeOrdinal -
                                     binding.traversalNodeOffset)},
         nodeOrdinal});
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Scheduled;
    scheduled = true;
  }
  return scheduled;
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

llvm::Error CgraTransportRuntime::acceptTransfers(
    const SpatialEventCoordinate &coordinate,
    llvm::ArrayRef<PendingTransfer> transfers) {
  if (transfers.empty())
    return llvm::Error::success();
  auto arrival = nextSpatialDelta(coordinate);
  if (!arrival)
    return arrival.takeError();
  if (transfers.size() >
      std::numeric_limits<std::uint64_t>::max() - inFlight_.size())
    return invalid("CGRA in-flight transport slot count exceeds u64");
  if (transfers.size() >
      std::numeric_limits<std::uint64_t>::max() - activeTransferCount_)
    return invalid("CGRA active transport count exceeds u64");

  llvm::SmallVector<std::uint64_t, 4> prospectiveSlots;
  prospectiveSlots.reserve(transfers.size());
  const std::size_t reused = std::min(transfers.size(), freeSlots_.size());
  for (std::size_t index = 0; index != reused; ++index)
    prospectiveSlots.push_back(freeSlots_[freeSlots_.size() - 1 - index]);
  for (std::size_t index = reused; index != transfers.size(); ++index)
    prospectiveSlots.push_back(inFlight_.size() + index - reused);

  llvm::SmallVector<PendingActionTransfer, 4> producedTransfers;
  producedTransfers.reserve(transfers.size());
  for (auto [transfer, slot] : llvm::zip(transfers, prospectiveSlots)) {
    if (!transfer.token || transfer.bindingOrdinal >= bindings_.size())
      return invalid("CGRA transport received a malformed source emission");
    const TransferBinding &binding = bindings_[transfer.bindingOrdinal];
    if (binding.requiresStorageTransport)
      return llvm::createStringError(
          std::errc::not_supported,
          "CGRA selected transfer requires typed traversal storage");
    producedTransfers.push_back({slot, transfer.bindingOrdinal});
  }

  auto requested =
      requestActions(producedTransfers, ActionStage::Produced, coordinate);
  if (!requested)
    return requested.takeError();

  llvm::SmallVector<std::uint64_t, 4> slots;
  slots.reserve(transfers.size());
  for (auto [transfer, expectedSlot] : llvm::zip(transfers, prospectiveSlots)) {
    const std::uint64_t slot =
        allocate(transfer.bindingOrdinal, transfer.occurrenceOrdinal,
                 std::move(*transfer.token));
    assert(slot == expectedSlot && "transport slot projection changed");
    slots.push_back(slot);
  }
  for (const CgraPhysicalLifecycleEvent &event : *requested)
    requestedEvents_.schedule(
        {{event.coordinate, event.actionOrdinal, event.occurrenceOrdinal,
          event.ownerEventOrdinal},
         0});
  for (std::uint64_t slot : slots) {
    const TransferBinding &binding = bindings_[inFlight_[slot].bindingOrdinal];
    if (binding.physicalUseCount == 0) {
      if (binding.traversalNodeCount == 0) {
        if (llvm::Error error = scheduleArrival(slot, *arrival))
          return error;
      } else {
        auto scheduled = scheduleReadyTraversals(slot, *arrival);
        if (!scheduled)
          return scheduled.takeError();
        if (!*scheduled)
          return invalid("CGRA traversal DAG has no ready root action");
      }
    }
  }
  return llvm::Error::success();
}

void CgraTransportRuntime::scheduleAt(
    std::uint64_t slot, const SpatialEventCoordinate &publicationCoordinate) {
  assert(slot < inFlight_.size() && inFlight_[slot].active &&
         "CGRA transport scheduled an inactive token");
  const InFlight &token = inFlight_[slot];
  events_.schedule({{publicationCoordinate, token.bindingOrdinal,
                     token.occurrenceOrdinal, 0},
                    slot});
}

llvm::Error CgraTransportRuntime::acceptActorEmissions(
    const SpatialEventCoordinate &coordinate,
    llvm::MutableArrayRef<CgraComputeActorEmission> emissions) {
  if (emissions.empty())
    return llvm::Error::success();
  llvm::SmallVector<PendingTransfer, 4> transfers;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueBindings;
  transfers.reserve(emissions.size());
  for (CgraComputeActorEmission &emission : emissions) {
    auto binding = actorSourceBindings_.find(
        {emission.actorPlanOrdinal, emission.resultOrdinal});
    if (binding == actorSourceBindings_.end())
      return invalid("CGRA actor emission has no selected transfer binding");
    if (bindings_[binding->second].active ||
        !uniqueBindings.insert(binding->second).second)
      return invalid("CGRA actor emission batch reuses an in-flight source");
    transfers.push_back(
        {binding->second, emission.occurrenceOrdinal, &emission.token});
  }
  return acceptTransfers(coordinate, transfers);
}

llvm::Error CgraTransportRuntime::acceptGraphIngressEmissions(
    const SpatialEventCoordinate &coordinate,
    llvm::MutableArrayRef<GraphIngressEmission> emissions) {
  if (emissions.empty())
    return llvm::Error::success();
  llvm::SmallVector<PendingTransfer, 4> transfers;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueBindings;
  transfers.reserve(emissions.size());
  for (GraphIngressEmission &emission : emissions) {
    auto binding = ingressSourceBindings_.find(emission.argumentOrdinal);
    if (binding == ingressSourceBindings_.end())
      return invalid("CGRA graph ingress has no selected transfer binding");
    if (bindings_[binding->second].active ||
        !uniqueBindings.insert(binding->second).second)
      return invalid("CGRA graph ingress batch reuses an in-flight source");
    transfers.push_back(
        {binding->second, emission.occurrenceOrdinal, &emission.token});
  }
  return acceptTransfers(coordinate, transfers);
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
  llvm::SmallVector<std::pair<std::uint64_t, std::uint64_t>, 8>
      newlyPermittedTraversals;

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
    case ActionStage::Consumed:
      permittedCount = &inFlight.consumedPermitted;
      retiredCount = &inFlight.consumedRetired;
      break;
    }
    const bool requiresCommit =
        plan_->physicalUseTimings[event.actionOrdinal].commitRank.has_value();
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
  llvm::sort(completions, [](const CgraTransportCompletion &lhs,
                             const CgraTransportCompletion &rhs) {
    return std::tie(lhs.actorPlanOrdinal, lhs.occurrenceOrdinal) <
           std::tie(rhs.actorPlanOrdinal, rhs.occurrenceOrdinal);
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
  if (binding.actorPlanOrdinal)
    completion = CgraTransportCompletion{*binding.actorPlanOrdinal,
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
      const TraversalNodeBinding &node = traversalNodes_[nodeOrdinal];
      if (nodeOrdinal < binding.traversalNodeOffset ||
          nodeOrdinal >=
              binding.traversalNodeOffset + binding.traversalNodeCount ||
          node.physicalUseOrdinal != event.order.structuralActionOrdinal ||
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
    if (llvm::Error error = schedulePublication(*slot, *publication))
      return error;
  }
  return llvm::Error::success();
}

} // namespace loom::sim::detail

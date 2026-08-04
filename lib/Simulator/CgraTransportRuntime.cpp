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
  bool requiresPhysicalTransport = false;
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
    SimulatorState &state, std::vector<TransferBinding> bindings,
    std::vector<SinkBinding> sinks,
    llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
        actorSourceBindings,
    llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings)
    : state_(&state), bindings_(std::move(bindings)), sinks_(std::move(sinks)),
      actorSourceBindings_(std::move(actorSourceBindings)),
      ingressSourceBindings_(std::move(ingressSourceBindings)),
      blocked_(bindings_.size()) {}

llvm::Expected<CgraTransportRuntime> CgraTransportRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
    SimulatorState &state) {
  auto resolvedGraph = dataflow.resolve(graph);
  if (!resolvedGraph)
    return resolvedGraph.takeError();
  auto graphOp = mlir::dyn_cast<::dataflow::GraphOp>(resolvedGraph->op);
  if (!graphOp)
    return invalid("CGRA transport graph reference is not a graph");
  std::map<RefBytes, BindingBuilder> builders;
  const auto addSink = [&](const auto &transfer,
                           const auto &sink) -> llvm::Error {
    auto key = dataflowBytes(dataflow, transfer.producer);
    if (!key)
      return key.takeError();
    auto [position, inserted] = builders.try_emplace(
        *key, BindingBuilder{transfer.producer, {}, false});
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
    auto [position, inserted] =
        builders.try_emplace(*key, BindingBuilder{route.producer, {}, false});
    (void)inserted;
    if (route.nodeOffset > plan.transport.routeNodes.size() ||
        route.nodeCount > plan.transport.routeNodes.size() - route.nodeOffset ||
        route.sinkOffset > plan.transport.routeSinks.size() ||
        route.sinkCount > plan.transport.routeSinks.size() - route.sinkOffset)
      return invalid("CGRA RouteTree execution slice is malformed");
    const auto requiresPhysical = [&](std::uint64_t traversal) {
      if (traversal == invalidCgraTransportOrdinal)
        return false;
      if (traversal >= plan.transport.traversals.size())
        return true;
      const CgraSelectedTraversalPlan &selected =
          plan.transport.traversals[traversal];
      return selected.storageKind != CgraTraversalStorageKind::None ||
             selected.impliedUseCount != 0;
    };
    position->second.requiresPhysicalTransport |=
        requiresPhysical(route.localTraversalOrdinal);
    for (const auto &node : llvm::ArrayRef(plan.transport.routeNodes)
                                .slice(route.nodeOffset, route.nodeCount))
      position->second.requiresPhysicalTransport |=
          requiresPhysical(node.incomingTraversalOrdinal);
    for (const auto &sink : llvm::ArrayRef(plan.transport.routeSinks)
                                .slice(route.sinkOffset, route.sinkCount)) {
      if (requiresPhysical(sink.localTraversalOrdinal))
        position->second.requiresPhysicalTransport = true;
      position->second.sinks.push_back(sink.sink);
    }
  }

  std::map<std::uint64_t, std::uint64_t> actorPlanByEntity;
  for (auto [ordinal, actor] : llvm::enumerate(plan.computeActors))
    if (actor.graph == graph &&
        !actorPlanByEntity.emplace(actor.actor.entity.value(), ordinal).second)
      return invalid("CGRA transport found duplicate compute actor bindings");

  std::vector<TransferBinding> bindings;
  std::vector<SinkBinding> sinks;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings;
  bindings.reserve(builders.size());
  for (auto &[key, builder] : builders) {
    (void)key;
    std::set<RefBytes> uniqueSinks;
    const std::uint64_t sinkOffset = sinks.size();
    for (const auto &sink : builder.sinks) {
      auto sinkKey = dataflowBytes(dataflow, sink);
      if (!sinkKey)
        return sinkKey.takeError();
      if (!uniqueSinks.insert(std::move(*sinkKey)).second)
        return invalid("CGRA transport contains a duplicate software sink");
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
        sinks.push_back({SinkKind::Channel, channel->second, {}});
      } else {
        auto observed = resolveObservation(
            execution, std::get<::dataflow::GraphEgressTokenRef>(sink));
        if (!observed)
          return observed.takeError();
        sinks.push_back({SinkKind::Observation, 0, *observed});
      }
    }
    if (builder.sinks.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA transport sink count exceeds u32");
    const std::uint64_t bindingOrdinal = bindings.size();
    bindings.push_back({builder.producer, sinkOffset,
                        static_cast<std::uint32_t>(builder.sinks.size()),
                        builder.requiresPhysicalTransport, false});
    if (const auto *producer =
            std::get_if<::dataflow::ActorTokenResultRef>(&builder.producer)) {
      auto actor = actorPlanByEntity.find(producer->actor.entity.value());
      if (actor == actorPlanByEntity.end())
        return invalid("CGRA transport producer has no compute actor binding");
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
  }
  return CgraTransportRuntime(state, std::move(bindings), std::move(sinks),
                              std::move(actorSourceBindings),
                              std::move(ingressSourceBindings));
}

std::uint64_t CgraTransportRuntime::allocate(std::uint64_t bindingOrdinal,
                                             std::uint64_t occurrenceOrdinal,
                                             Token token) {
  assert(bindingOrdinal < bindings_.size() &&
         !bindings_[bindingOrdinal].active &&
         "CGRA transport allocation requires a validated source");
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
  bindings_[bindingOrdinal].active = true;
  return slot;
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
  auto publication = nextSpatialDelta(coordinate);
  if (!publication)
    return publication.takeError();

  llvm::SmallVector<std::uint64_t, 4> bindingOrdinals;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueBindings;
  bindingOrdinals.reserve(emissions.size());
  for (const CgraComputeActorEmission &emission : emissions) {
    auto binding = actorSourceBindings_.find(
        {emission.actorPlanOrdinal, emission.resultOrdinal});
    if (binding == actorSourceBindings_.end())
      return invalid("CGRA actor emission has no selected transfer binding");
    if (bindings_[binding->second].active ||
        !uniqueBindings.insert(binding->second).second)
      return invalid("CGRA actor emission batch reuses an in-flight source");
    if (bindings_[binding->second].requiresPhysicalTransport)
      return llvm::createStringError(
          std::errc::not_supported,
          "CGRA selected transfer requires physical resource coordination");
    bindingOrdinals.push_back(binding->second);
  }

  llvm::SmallVector<std::uint64_t, 4> slots;
  slots.reserve(emissions.size());
  for (auto [emission, binding] : llvm::zip(emissions, bindingOrdinals)) {
    slots.push_back(allocate(binding, emission.occurrenceOrdinal,
                             std::move(emission.token)));
  }
  for (std::uint64_t slot : slots)
    scheduleAt(slot, *publication);
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::acceptGraphIngressEmissions(
    const SpatialEventCoordinate &coordinate,
    llvm::MutableArrayRef<GraphIngressEmission> emissions) {
  if (emissions.empty())
    return llvm::Error::success();
  auto publication = nextSpatialDelta(coordinate);
  if (!publication)
    return publication.takeError();

  llvm::SmallVector<std::uint64_t, 4> bindingOrdinals;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueBindings;
  bindingOrdinals.reserve(emissions.size());
  for (const GraphIngressEmission &emission : emissions) {
    auto binding = ingressSourceBindings_.find(emission.argumentOrdinal);
    if (binding == ingressSourceBindings_.end())
      return invalid("CGRA graph ingress has no selected transfer binding");
    if (bindings_[binding->second].active ||
        !uniqueBindings.insert(binding->second).second)
      return invalid("CGRA graph ingress batch reuses an in-flight source");
    if (bindings_[binding->second].requiresPhysicalTransport)
      return llvm::createStringError(
          std::errc::not_supported,
          "CGRA selected ingress requires physical resource coordination");
    bindingOrdinals.push_back(binding->second);
  }

  llvm::SmallVector<std::uint64_t, 4> slots;
  slots.reserve(emissions.size());
  for (auto [emission, binding] : llvm::zip(emissions, bindingOrdinals))
    slots.push_back(allocate(binding, emission.occurrenceOrdinal,
                             std::move(emission.token)));
  for (std::uint64_t slot : slots)
    scheduleAt(slot, *publication);
  return llvm::Error::success();
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
  release(slot);
}

void CgraTransportRuntime::release(std::uint64_t slot) {
  InFlight &inFlight = inFlight_[slot];
  bindings_[inFlight.bindingOrdinal].active = false;
  blocked_.reset(inFlight.bindingOrdinal);
  inFlight.active = false;
  freeSlots_.push_back(slot);
}

llvm::Expected<std::optional<CgraTransportFrame>>
CgraTransportRuntime::advance() {
  auto eventFrame = events_.popNextFrame();
  if (!eventFrame)
    return eventFrame.takeError();
  if (!*eventFrame)
    return std::optional<CgraTransportFrame>{};
  CgraTransportFrame frame{(**eventFrame).coordinate, {}, {}};
  for (const CgraScheduledEvent &event : (**eventFrame).events) {
    if (event.payload >= inFlight_.size() || !inFlight_[event.payload].active)
      return invalid("CGRA transport event names an inactive token");
    InFlight &inFlight = inFlight_[event.payload];
    if (inFlight.bindingOrdinal != event.order.structuralActionOrdinal ||
        inFlight.occurrenceOrdinal != event.order.occurrenceOrdinal)
      return invalid("CGRA transport event key disagrees with its token");
    TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    if (!canPublish(binding)) {
      blocked_.set(inFlight.bindingOrdinal);
      frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
      continue;
    }
    publish(event.payload, frame);
  }
  return std::optional<CgraTransportFrame>(std::move(frame));
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
    scheduleAt(*slot, *publication);
  }
  return llvm::Error::success();
}

} // namespace loom::sim::detail

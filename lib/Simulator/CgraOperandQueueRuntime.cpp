#include "CgraTransportRuntime.h"

#include "CgraFabricActivityRuntime.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Error CgraTransportRuntime::observeOperandQueueActivity(
    std::uint64_t queueOrdinal, const SpatialEventCoordinate &coordinate) {
  auto *activity = physical_->activity();
  if (!activity)
    return llvm::Error::success();
  const auto &queue = operandQueues_[queueOrdinal];
  const auto &buffer = graph_.operandBuffers[queue.binding.bufferBinding];
  const auto &unit = operandQueueUnits_[queue.binding.unitBinding];
  return activity->observeOperandQueue(
      buffer.pe, buffer.contract, queue.binding.contractQueue,
      unit.binding.allocationUnit, queue.occupancy, unit.occupancy, coordinate);
}

llvm::Error CgraTransportRuntime::initializeActivity(
    const SpatialEventCoordinate &coordinate) {
  auto *activity = physical_->activity();
  if (!activity)
    return llvm::Error::success();
  for (std::uint64_t ordinal = 0; ordinal != operandQueues_.size(); ++ordinal)
    if (llvm::Error error = observeOperandQueueActivity(ordinal, coordinate))
      return error;
  for (std::uint64_t ordinal = 0; ordinal != storages_.size(); ++ordinal)
    if (llvm::Error error = activity->observeTraversalStorage(
            ordinal, storages_[ordinal].queue.occupancy(), coordinate))
      return error;
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::beginOperandQueueCycle(
    const SpatialEventCoordinate &coordinate) {
  const SpatialEventCoordinate incoming{coordinate.referenceCycle, 0};
  for (OperandQueueUnitState &unit : operandQueueUnits_) {
    if (unit.admissionCycle) {
      const SpatialEventCoordinate active{*unit.admissionCycle, 0};
      const int order = compareSpatialEventCoordinates(incoming, active);
      if (order < 0)
        return invalid("CGRA PE operand admission cycle moved backward");
      if (order == 0)
        continue;
    }
    if (unit.occupancy > unit.binding.capacity ||
        unit.reservations > unit.binding.capacity - unit.occupancy)
      return invalid("CGRA PE operand allocation-unit occupancy is invalid");
    unit.admissionCycle = coordinate.referenceCycle;
    unit.admissionCredits =
        unit.binding.capacity - unit.occupancy - unit.reservations;
  }
  return llvm::Error::success();
}

llvm::Expected<CgraTransportRuntime::OperandIngressAdmission>
CgraTransportRuntime::operandIngressAdmissionPriority(
    std::uint64_t slot, std::uint64_t publicationBinding) const {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA operand ingress priority names an inactive token");
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = graph_.bindings[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA operand ingress priority names another publication");
  const PublicationBinding &publication =
      graph_.publications[publicationBinding];

  struct BufferQuery final {
    std::uint64_t buffer = invalidCgraTransportOrdinal;
    llvm::SmallVector<std::uint32_t, 8> matched;
    llvm::SmallVector<std::uint32_t, 8> required;
  };
  llvm::SmallVector<BufferQuery, 2> queries;
  OperandIngressAdmission result;
  for (std::uint32_t localSink :
       llvm::ArrayRef(graph_.publicationSinks)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA operand ingress priority has an unknown sink");
    const SinkBinding &sink = graph_.sinks[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.operandQueueBinding >= operandQueues_.size() ||
        sink.operandActivationOrdinal >=
            plan_->transport.operandQueueActivations.size())
      return invalid("CGRA operand ingress priority has an invalid queue");
    const OperandQueueState &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.binding.bufferBinding >= graph_.operandBuffers.size())
      return invalid("CGRA operand ingress priority lost its Fabric owner");
    auto query = llvm::find_if(queries, [&](const auto &candidate) {
      return candidate.buffer == queue.binding.bufferBinding;
    });
    if (query == queries.end()) {
      queries.push_back({queue.binding.bufferBinding, {}, {}});
      query = queries.end() - 1;
    }
    query->matched.push_back(queue.binding.contractQueue);
    const llvm::APInt &tag =
        plan_->transport.operandQueueActivations[sink.operandActivationOrdinal]
            .tag;
    const auto pairing = llvm::find_if(
        plan_->transport.operandQueueProgress.pairings,
        [&](const auto &candidate) {
          return candidate.key.context == queue.binding.queue.context &&
                 candidate.key.fu == queue.binding.fu &&
                 candidate.key.tag.getBitWidth() == tag.getBitWidth() &&
                 candidate.key.tag == tag;
        });
    if (pairing == plan_->transport.operandQueueProgress.pairings.end())
      return invalid("CGRA operand ingress priority has no PairingKey");
    if (!llvm::is_contained(result.pairings, pairing->key))
      result.pairings.push_back(pairing->key);
    const OperandBufferBinding &buffer =
        graph_.operandBuffers[queue.binding.bufferBinding];
    for (std::uint32_t role : pairing->requiredInputRoles) {
      const ::fabric::LogicalOperandQueueKey requiredKey{
          queue.binding.queue.context, queue.binding.queue.fuOccurrence, role};
      const auto required =
          llvm::lower_bound(buffer.contract.logicalQueues(), requiredKey);
      if (required == buffer.contract.logicalQueues().end() ||
          *required != requiredKey)
        return invalid("CGRA operand ingress priority lost a required "
                       "QueueKey");
      const std::uint32_t contractQueue = static_cast<std::uint32_t>(
          std::distance(buffer.contract.logicalQueues().begin(), required));
      if (buffer.runtimeQueues[contractQueue] == invalidCgraTransportOrdinal)
        return invalid("CGRA operand ingress priority has no runtime binding "
                       "for a required QueueKey");
      query->required.push_back(contractQueue);
    }
  }

  for (BufferQuery &query : queries) {
    const std::uint64_t bufferOrdinal = query.buffer;
    llvm::sort(query.matched);
    query.matched.erase(std::unique(query.matched.begin(), query.matched.end()),
                        query.matched.end());
    llvm::sort(query.required);
    query.required.erase(
        std::unique(query.required.begin(), query.required.end()),
        query.required.end());
    const OperandBufferBinding &buffer = graph_.operandBuffers[bufferOrdinal];
    llvm::SmallVector<::fabric::OperandQueueCycleObservation, 32> observations(
        buffer.contract.logicalQueues().size(),
        {false, ::fabric::CapacityUnits(0)});
    for (std::uint32_t queue = 0; queue != observations.size(); ++queue) {
      const std::uint32_t allocationUnit =
          buffer.contract.allocationUnitOf(queue);
      const std::uint64_t unit = buffer.runtimeUnits[allocationUnit];
      if (unit != invalidCgraTransportOrdinal) {
        if (unit >= operandQueueUnits_.size())
          return invalid("CGRA operand ingress priority has an invalid dense "
                         "unit binding");
        observations[queue].allocationUnitOccupancy =
            ::fabric::CapacityUnits(operandQueueUnits_[unit].occupancy);
      }
    }
    for (std::uint32_t contractQueue = 0;
         contractQueue != buffer.runtimeQueues.size(); ++contractQueue) {
      const std::uint64_t runtimeQueue = buffer.runtimeQueues[contractQueue];
      if (runtimeQueue == invalidCgraTransportOrdinal)
        continue;
      if (runtimeQueue >= operandQueues_.size())
        return invalid("CGRA operand ingress priority has an invalid dense "
                       "queue binding");
      observations[contractQueue].headPresent =
          operandQueues_[runtimeQueue].occupancy != 0;
    }
    auto priority = buffer.contract.ingressAdmissionPriority(
        query.matched, query.required, observations);
    if (!priority)
      return priority.takeError();
    if (static_cast<std::uint8_t>(*priority) >
        static_cast<std::uint8_t>(result.priority))
      result.priority = *priority;
  }
  return result;
}

llvm::Expected<bool> CgraTransportRuntime::reserveOperandQueueCapacity(
    std::uint64_t slot, std::uint64_t publicationBinding,
    const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA PE operand reservation names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = graph_.bindings[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA PE operand reservation names another publication");
  InFlight::PublicationState &state =
      inFlight.publications[publicationBinding - binding.publicationOffset];
  if (state.capacityReserved)
    return invalid("CGRA PE operand capacity was reserved twice");
  const PublicationBinding &publication =
      graph_.publications[publicationBinding];

  llvm::SmallVector<std::uint64_t, 4> units;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueQueues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (std::uint32_t localSink :
       llvm::ArrayRef(graph_.publicationSinks)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA PE operand publication names an unknown sink");
    const SinkBinding &sink = graph_.sinks[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.kind != SinkKind::Channel ||
        sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand sink has an invalid queue binding");
    const OperandQueueState &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.binding.unitBinding >= operandQueueUnits_.size() ||
        llvm::none_of(queue.binding.consumers, [&](const auto &consumer) {
          return consumer.channel == sink.channel;
        }))
      return invalid("CGRA PE operand queue state diverged from its channel");
    for (const auto &consumer : queue.binding.consumers)
      if (consumer.channel >= state_->channelSlots.size() ||
          state_->channelSlots[consumer.channel].ready.size() !=
              queue.occupancy)
        return invalid(
            "CGRA PE operand broadcast consumer state diverged from its "
            "queue");
    if (!uniqueQueues.insert(sink.operandQueueBinding).second)
      continue;
    if (!uniqueUnits.insert(queue.binding.unitBinding).second)
      return invalid("CGRA PE operand activation repeats an allocation unit");
    units.push_back(queue.binding.unitBinding);
  }

  if (!units.empty())
    if (llvm::Error error = beginOperandQueueCycle(coordinate))
      return std::move(error);
  for (std::uint64_t unitOrdinal : units) {
    const OperandQueueUnitState &unit = operandQueueUnits_[unitOrdinal];
    if (unit.occupancy > unit.binding.capacity ||
        unit.reservations > unit.binding.capacity - unit.occupancy)
      return invalid("CGRA PE operand allocation-unit occupancy is invalid");
    if (unit.admissionCredits == 0) {
      state.capacityBlocked = true;
      return false;
    }
  }
  for (std::uint64_t unitOrdinal : units) {
    --operandQueueUnits_[unitOrdinal].admissionCredits;
    ++operandQueueUnits_[unitOrdinal].reservations;
  }
  state.capacityReserved = !units.empty();
  state.capacityBlocked = false;
  return true;
}

llvm::Error CgraTransportRuntime::commitOperandQueueEnqueue(
    std::uint64_t slot, std::uint64_t publicationBinding,
    const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA PE operand enqueue names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = graph_.bindings[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA PE operand enqueue names another publication");
  InFlight::PublicationState &state =
      inFlight.publications[publicationBinding - binding.publicationOffset];
  const PublicationBinding &publication =
      graph_.publications[publicationBinding];
  llvm::SmallVector<std::uint64_t, 4> queues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueQueues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (std::uint32_t localSink :
       llvm::ArrayRef(graph_.publicationSinks)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA PE operand publication names an unknown sink");
    const SinkBinding &sink = graph_.sinks[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand enqueue has an invalid queue binding");
    const OperandQueueState &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.binding.unitBinding >= operandQueueUnits_.size() ||
        llvm::none_of(queue.binding.consumers, [&](const auto &consumer) {
          return consumer.channel == sink.channel;
        }))
      return invalid("CGRA PE operand enqueue found divergent queue state");
    for (const auto &consumer : queue.binding.consumers)
      if (consumer.channel >= state_->channelSlots.size() ||
          state_->channelSlots[consumer.channel].ready.size() !=
              queue.occupancy)
        return invalid(
            "CGRA PE operand enqueue found divergent broadcast state");
    if (!uniqueQueues.insert(sink.operandQueueBinding).second)
      continue;
    if (!uniqueUnits.insert(queue.binding.unitBinding).second)
      return invalid("CGRA PE operand enqueue repeats an allocation unit");
    const OperandQueueUnitState &unit =
        operandQueueUnits_[queue.binding.unitBinding];
    if (unit.reservations == 0 || unit.occupancy >= unit.binding.capacity)
      return invalid("CGRA PE operand enqueue has no reserved capacity");
    queues.push_back(sink.operandQueueBinding);
  }
  if (queues.empty()) {
    if (state.capacityReserved)
      return invalid("CGRA non-queue transfer retained operand capacity");
    return llvm::Error::success();
  }
  if (!state.capacityReserved)
    return invalid("CGRA PE operand enqueue was not atomically reserved");

  for (std::uint64_t queueOrdinal : queues) {
    OperandQueueState &queue = operandQueues_[queueOrdinal];
    OperandQueueUnitState &unit = operandQueueUnits_[queue.binding.unitBinding];
    --unit.reservations;
    ++unit.occupancy;
    ++queue.occupancy;
    if (llvm::Error error =
            observeOperandQueueActivity(queueOrdinal, coordinate))
      return error;
    std::optional<llvm::APInt> tag;
    for (std::uint32_t localSink :
         llvm::ArrayRef(graph_.publicationSinks)
             .slice(publication.sinkOffset, publication.sinkCount)) {
      const SinkBinding &sink = graph_.sinks[binding.sinkOffset + localSink];
      if (sink.operandQueueBinding != queueOrdinal)
        continue;
      if (sink.operandActivationOrdinal >=
          plan_->transport.operandQueueActivations.size())
        return invalid("CGRA PE operand enqueue has no activation tag");
      const llvm::APInt &candidateTag =
          plan_->transport
              .operandQueueActivations[sink.operandActivationOrdinal]
              .tag;
      if (tag && *tag != candidateTag)
        return invalid("CGRA PE operand enqueue has conflicting queue tags");
      tag = candidateTag;
    }
    if (!tag)
      return invalid("CGRA PE operand enqueue has no queue tag witness");
    queue.entries.push_back({slot, inFlight.occurrenceOrdinal,
                             inFlight.producerSequenceOrdinal, *tag});
  }
  state.capacityReserved = false;
  return llvm::Error::success();
}

llvm::Expected<std::vector<CgraTransportCompletion>>
CgraTransportRuntime::acceptActorCommits(
    llvm::ArrayRef<CgraActorLifecycleEvent> events) {
  struct Dequeue final {
    std::uint64_t queue = 0;
    std::uint64_t unit = 0;
  };
  llvm::SmallVector<Dequeue, 8> dequeues;
  llvm::SmallDenseSet<std::pair<std::uint64_t, unsigned>, 8> consumedInputs;
  llvm::SmallDenseSet<std::uint64_t, 8> touchedQueues;
  llvm::SmallDenseSet<std::uint64_t, 8> frameUnits;
  llvm::SmallVector<std::uint64_t, 8> sourceReservations;
  llvm::SmallDenseSet<std::uint64_t, 8> touchedSourceBindings;
  if (!events.empty()) {
    const auto &cycle = events.front().coordinate.referenceCycle;
    for (const CgraActorLifecycleEvent &event : events)
      if (event.coordinate.referenceCycle != cycle)
        return invalid("CGRA PE operand dequeue batch spans clock cycles");
    if (llvm::Error error = beginOperandQueueCycle(events.front().coordinate))
      return error;
  }
  for (const CgraActorLifecycleEvent &event : events) {
    if (event.kind != CgraActorLifecycleKind::Committed ||
        event.semanticActorOrdinal >= state_->execution->actorPlans.size())
      return invalid("CGRA PE operand dequeue has an invalid actor commit");
    const ActorExecutionPlan &actor =
        state_->execution->actorPlans[event.semanticActorOrdinal];
    const auto transition =
        llvm::find_if(actor.handshakeCases, [&](const auto &candidate) {
          return candidate.ordinal == event.transitionCaseOrdinal;
        });
    if (transition == actor.handshakeCases.end())
      return invalid("CGRA PE operand dequeue names an unknown transition");
    for (std::uint32_t result : transition->activeResults) {
      const auto binding =
          graph_.actorSourceBindings.find({event.semanticActorOrdinal, result});
      if (binding == graph_.actorSourceBindings.end())
        continue;
      if (binding->second >= graph_.bindings.size())
        return invalid("CGRA actor commit has an invalid transport source");
      if (producerStates_[binding->second].sourceReserved ||
          producerStates_[binding->second].producerPending)
        return invalid(llvm::Twine("CGRA actor ") +
                       llvm::Twine(event.semanticActorOrdinal) +
                       " occurrence " + llvm::Twine(event.occurrenceOrdinal) +
                       " result " + llvm::Twine(result) +
                       " commits through a busy transport binding " +
                       llvm::Twine(binding->second));
      if (!touchedSourceBindings.insert(binding->second).second)
        return invalid("CGRA actor commit batch repeats a transport source");
      sourceReservations.push_back(binding->second);
    }
    for (std::uint32_t input : transition->consumedInputs) {
      if (!consumedInputs.insert({event.semanticActorOrdinal, input}).second)
        return invalid("CGRA actor commit repeats a consumed input");
      auto found = graph_.actorInputQueueBindings.find(
          {event.semanticActorOrdinal, input});
      if (found == graph_.actorInputQueueBindings.end())
        continue;
      if (found->second >= operandQueues_.size())
        return invalid("CGRA PE operand dequeue has an invalid queue binding");
      touchedQueues.insert(found->second);
    }
  }
  llvm::SmallVector<std::uint64_t, 8> orderedQueues(touchedQueues.begin(),
                                                    touchedQueues.end());
  llvm::sort(orderedQueues);
  for (std::uint64_t queueOrdinal : orderedQueues) {
    if (queueOrdinal >= operandQueues_.size())
      return invalid("CGRA PE operand dequeue has an invalid queue binding");
    const OperandQueueState &queue = operandQueues_[queueOrdinal];
    if (queue.binding.unitBinding >= operandQueueUnits_.size() ||
        queue.occupancy == 0 ||
        operandQueueUnits_[queue.binding.unitBinding].occupancy == 0 ||
        queue.binding.consumers.empty())
      return invalid("CGRA PE operand dequeue underflows its queue");
    for (const auto &consumer : queue.binding.consumers) {
      if (!consumedInputs.contains(
              {consumer.semanticActorOrdinal, consumer.inputOrdinal})) {
        std::string diagnostic =
            "CGRA PE operand broadcast queue " + std::to_string(queueOrdinal) +
            " omitted actor " + std::to_string(consumer.semanticActorOrdinal) +
            " input " + std::to_string(consumer.inputOrdinal) +
            "; committed actors";
        for (const CgraActorLifecycleEvent &event : events)
          diagnostic += " " + std::to_string(event.semanticActorOrdinal);
        diagnostic += "; queue consumers";
        for (const auto &member : queue.binding.consumers)
          diagnostic += " " + std::to_string(member.semanticActorOrdinal) +
                        ":" + std::to_string(member.inputOrdinal);
        return invalid(diagnostic);
      }
      if (consumer.channel >= state_->channelSlots.size())
        return invalid("CGRA PE operand consumer channel is out of range");
      const std::size_t channelOccupancy =
          state_->channelSlots[consumer.channel].ready.size();
      if (channelOccupancy == std::numeric_limits<std::size_t>::max() ||
          channelOccupancy + 1 != queue.occupancy)
        return invalid(
            "CGRA PE operand dequeue diverged from a broadcast consumer");
    }
    if (!frameUnits.insert(queue.binding.unitBinding).second)
      return invalid("CGRA PE operand dequeue service committed twice");
    dequeues.push_back({queueOrdinal, queue.binding.unitBinding});
  }
  // Published unbuffered heads remain owned by their active transfer until
  // the exact receiving actor consumes them. Derive acknowledgements from
  // those existing owners rather than adding another token or credit queue.
  std::vector<std::pair<std::uint64_t, llvm::SmallVector<std::uint32_t, 4>>>
      handoffs;
  for (auto [slot, transfer] : llvm::enumerate(inFlight_)) {
    if (!transfer.active)
      continue;
    const TransferBinding &binding = graph_.bindings[transfer.bindingOrdinal];
    llvm::SmallVector<std::uint32_t, 4> consumedSinks;
    for (std::uint32_t local = 0; local != binding.sinkCount; ++local) {
      if (!transfer.publishedSinks[local] || transfer.acceptedSinks[local])
        continue;
      const SinkBinding &sink = graph_.sinks[binding.sinkOffset + local];
      if (sink.kind != SinkKind::Channel ||
          sink.operandQueueBinding != invalidCgraTransportOrdinal ||
          !consumedInputs.contains({sink.semanticActorOrdinal,
                                    sink.inputOrdinal}))
        continue;
      if (!state_->channelSlots[sink.channel].ready.empty())
        return invalid("CGRA unbuffered handoff did not consume its head");
      consumedSinks.push_back(local);
    }
    if (!consumedSinks.empty())
      handoffs.emplace_back(slot, std::move(consumedSinks));
  }
  for (const Dequeue &dequeue : dequeues) {
    OperandQueueState &queue = operandQueues_[dequeue.queue];
    if (queue.entries.size() != queue.occupancy || queue.entries.empty())
      return invalid("CGRA PE operand queue head witness diverged from "
                     "occupancy");
    queue.entries.pop_front();
    --queue.occupancy;
    --operandQueueUnits_[dequeue.unit].occupancy;
    if (llvm::Error error = observeOperandQueueActivity(
            dequeue.queue, events.front().coordinate))
      return error;
  }
  for (std::uint64_t binding : sourceReservations)
    producerStates_[binding].sourceReserved = true;
  std::vector<CgraTransportCompletion> completions;
  for (const auto &[slot, sinks] : handoffs) {
    if (llvm::Error error = acceptDurableSinks(slot, sinks))
      return std::move(error);
    auto completed = maybeCompleteProducer(slot);
    if (!completed)
      return completed.takeError();
    if (*completed)
      completions.push_back(**completed);
    if (auto released = maybeRelease(slot))
      completions.push_back(*released);
  }
  return completions;
}

} // namespace loom::sim::detail

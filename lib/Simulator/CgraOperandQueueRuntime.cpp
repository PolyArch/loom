#include "CgraTransportRuntime.h"

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

llvm::Error CgraTransportRuntime::beginOperandQueueCycle(
    const SpatialEventCoordinate &coordinate) {
  const SpatialEventCoordinate incoming{coordinate.referenceCycle, 0};
  for (OperandQueueUnitBinding &unit : operandQueueUnits_) {
    if (unit.admissionCycle) {
      const SpatialEventCoordinate active{*unit.admissionCycle, 0};
      const int order = compareSpatialEventCoordinates(incoming, active);
      if (order < 0)
        return invalid("CGRA PE operand admission cycle moved backward");
      if (order == 0)
        continue;
    }
    if (unit.occupancy > unit.capacity ||
        unit.reservations > unit.capacity - unit.occupancy)
      return invalid("CGRA PE operand allocation-unit occupancy is invalid");
    unit.admissionCycle = coordinate.referenceCycle;
    unit.admissionCredits = unit.capacity - unit.occupancy - unit.reservations;
  }
  return llvm::Error::success();
}

llvm::Expected<CgraTransportRuntime::OperandIngressAdmission>
CgraTransportRuntime::operandIngressAdmissionPriority(
    std::uint64_t slot, std::uint64_t publicationBinding) const {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA operand ingress priority names an inactive token");
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA operand ingress priority names another publication");
  const PublicationBinding &publication = publications_[publicationBinding];

  struct BufferQuery final {
    std::uint64_t buffer = invalidCgraTransportOrdinal;
    llvm::SmallVector<std::uint32_t, 8> matched;
    llvm::SmallVector<std::uint32_t, 8> required;
  };
  llvm::SmallVector<BufferQuery, 2> queries;
  OperandIngressAdmission result;
  for (std::uint32_t localSink :
       llvm::ArrayRef(publicationSinks_)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA operand ingress priority has an unknown sink");
    const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.operandQueueBinding >= operandQueues_.size() ||
        sink.operandActivationOrdinal >=
            plan_->transport.operandQueueActivations.size())
      return invalid("CGRA operand ingress priority has an invalid queue");
    const OperandQueueBinding &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.bufferBinding >= operandBuffers_.size())
      return invalid("CGRA operand ingress priority lost its Fabric owner");
    auto query = llvm::find_if(queries, [&](const auto &candidate) {
      return candidate.buffer == queue.bufferBinding;
    });
    if (query == queries.end()) {
      queries.push_back({queue.bufferBinding, {}, {}});
      query = queries.end() - 1;
    }
    query->matched.push_back(queue.contractQueue);
    const llvm::APInt &tag =
        plan_->transport.operandQueueActivations[sink.operandActivationOrdinal]
            .tag;
    const auto pairing = llvm::find_if(
        plan_->transport.operandQueueProgress.pairings,
        [&](const auto &candidate) {
          return candidate.key.context == queue.queue.context &&
                 candidate.key.fu == queue.fu &&
                 candidate.key.tag.getBitWidth() == tag.getBitWidth() &&
                 candidate.key.tag == tag;
        });
    if (pairing == plan_->transport.operandQueueProgress.pairings.end())
      return invalid("CGRA operand ingress priority has no PairingKey");
    if (!llvm::is_contained(result.pairings, pairing->key))
      result.pairings.push_back(pairing->key);
    const OperandBufferBinding &buffer = operandBuffers_[queue.bufferBinding];
    for (std::uint32_t role : pairing->requiredInputRoles) {
      const ::fabric::LogicalOperandQueueKey requiredKey{
          queue.queue.context, queue.queue.fuOccurrence, role};
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
    const OperandBufferBinding &buffer = operandBuffers_[bufferOrdinal];
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
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA PE operand reservation names another publication");
  InFlight::PublicationState &state =
      inFlight.publications[publicationBinding - binding.publicationOffset];
  if (state.capacityReserved)
    return invalid("CGRA PE operand capacity was reserved twice");
  const PublicationBinding &publication = publications_[publicationBinding];

  llvm::SmallVector<std::uint64_t, 4> units;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueQueues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (std::uint32_t localSink :
       llvm::ArrayRef(publicationSinks_)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA PE operand publication names an unknown sink");
    const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.kind != SinkKind::Channel ||
        sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand sink has an invalid queue binding");
    const OperandQueueBinding &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.unitBinding >= operandQueueUnits_.size() ||
        llvm::none_of(queue.consumers, [&](const auto &consumer) {
          return consumer.channel == sink.channel;
        }))
      return invalid("CGRA PE operand queue state diverged from its channel");
    for (const auto &consumer : queue.consumers)
      if (consumer.channel >= state_->channelSlots.size() ||
          state_->channelSlots[consumer.channel].ready.size() !=
              queue.occupancy)
        return invalid(
            "CGRA PE operand broadcast consumer state diverged from its "
            "queue");
    if (!uniqueQueues.insert(sink.operandQueueBinding).second)
      continue;
    if (!uniqueUnits.insert(queue.unitBinding).second)
      return invalid("CGRA PE operand activation repeats an allocation unit");
    units.push_back(queue.unitBinding);
  }

  if (!units.empty())
    if (llvm::Error error = beginOperandQueueCycle(coordinate))
      return std::move(error);
  for (std::uint64_t unitOrdinal : units) {
    const OperandQueueUnitBinding &unit = operandQueueUnits_[unitOrdinal];
    if (unit.occupancy > unit.capacity ||
        unit.reservations > unit.capacity - unit.occupancy)
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
    std::uint64_t slot, std::uint64_t publicationBinding) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA PE operand enqueue names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA PE operand enqueue names another publication");
  InFlight::PublicationState &state =
      inFlight.publications[publicationBinding - binding.publicationOffset];
  const PublicationBinding &publication = publications_[publicationBinding];
  llvm::SmallVector<std::uint64_t, 4> queues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueQueues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (std::uint32_t localSink :
       llvm::ArrayRef(publicationSinks_)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount)
      return invalid("CGRA PE operand publication names an unknown sink");
    const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand enqueue has an invalid queue binding");
    const OperandQueueBinding &queue = operandQueues_[sink.operandQueueBinding];
    if (queue.unitBinding >= operandQueueUnits_.size() ||
        llvm::none_of(queue.consumers, [&](const auto &consumer) {
          return consumer.channel == sink.channel;
        }))
      return invalid("CGRA PE operand enqueue found divergent queue state");
    for (const auto &consumer : queue.consumers)
      if (consumer.channel >= state_->channelSlots.size() ||
          state_->channelSlots[consumer.channel].ready.size() !=
              queue.occupancy)
        return invalid(
            "CGRA PE operand enqueue found divergent broadcast state");
    if (!uniqueQueues.insert(sink.operandQueueBinding).second)
      continue;
    if (!uniqueUnits.insert(queue.unitBinding).second)
      return invalid("CGRA PE operand enqueue repeats an allocation unit");
    const OperandQueueUnitBinding &unit = operandQueueUnits_[queue.unitBinding];
    if (unit.reservations == 0 || unit.occupancy >= unit.capacity)
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
    OperandQueueBinding &queue = operandQueues_[queueOrdinal];
    OperandQueueUnitBinding &unit = operandQueueUnits_[queue.unitBinding];
    --unit.reservations;
    ++unit.occupancy;
    ++queue.occupancy;
    std::optional<llvm::APInt> tag;
    for (std::uint32_t localSink :
         llvm::ArrayRef(publicationSinks_)
             .slice(publication.sinkOffset, publication.sinkCount)) {
      const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
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

llvm::Error CgraTransportRuntime::acceptActorCommits(
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
          actorSourceBindings_.find({event.semanticActorOrdinal, result});
      if (binding == actorSourceBindings_.end())
        continue;
      if (binding->second >= bindings_.size())
        return invalid("CGRA actor commit has an invalid transport source");
      if (bindings_[binding->second].sourceReserved ||
          bindings_[binding->second].active)
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
      auto found =
          actorInputQueueBindings_.find({event.semanticActorOrdinal, input});
      if (found == actorInputQueueBindings_.end())
        continue;
      if (found->second >= operandQueues_.size())
        return invalid("CGRA PE operand dequeue has an invalid queue binding");
      if (!consumedInputs.insert({event.semanticActorOrdinal, input}).second)
        return invalid("CGRA PE operand dequeue repeats an actor input");
      touchedQueues.insert(found->second);
    }
  }
  llvm::SmallVector<std::uint64_t, 8> orderedQueues(touchedQueues.begin(),
                                                    touchedQueues.end());
  llvm::sort(orderedQueues);
  for (std::uint64_t queueOrdinal : orderedQueues) {
    if (queueOrdinal >= operandQueues_.size())
      return invalid("CGRA PE operand dequeue has an invalid queue binding");
    const OperandQueueBinding &queue = operandQueues_[queueOrdinal];
    if (queue.unitBinding >= operandQueueUnits_.size() ||
        queue.occupancy == 0 ||
        operandQueueUnits_[queue.unitBinding].occupancy == 0 ||
        queue.consumers.empty())
      return invalid("CGRA PE operand dequeue underflows its queue");
    for (const auto &consumer : queue.consumers) {
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
        for (const auto &member : queue.consumers)
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
    if (!frameUnits.insert(queue.unitBinding).second)
      return invalid("CGRA PE operand dequeue service committed twice");
    dequeues.push_back({queueOrdinal, queue.unitBinding});
  }
  for (const Dequeue &dequeue : dequeues) {
    OperandQueueBinding &queue = operandQueues_[dequeue.queue];
    if (queue.entries.size() != queue.occupancy || queue.entries.empty())
      return invalid("CGRA PE operand queue head witness diverged from "
                     "occupancy");
    queue.entries.pop_front();
    --queue.occupancy;
    --operandQueueUnits_[dequeue.unit].occupancy;
  }
  for (std::uint64_t binding : sourceReservations)
    bindings_[binding].sourceReserved = true;
  return llvm::Error::success();
}

} // namespace loom::sim::detail

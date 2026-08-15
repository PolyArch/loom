#include "CgraTransportRuntime.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<bool>
CgraTransportRuntime::reserveOperandQueueCapacity(std::uint64_t slot) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA PE operand reservation names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  if (inFlight.operandCapacityReserved)
    return invalid("CGRA PE operand capacity was reserved twice");
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];

  llvm::SmallVector<std::uint64_t, 4> units;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (const SinkBinding &sink :
       llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount)) {
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.kind != SinkKind::Channel ||
        sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand sink has an invalid queue binding");
    const OperandQueueBinding &queue =
        operandQueues_[sink.operandQueueBinding];
    if (queue.channel != sink.channel ||
        queue.unitBinding >= operandQueueUnits_.size() ||
        state_->channelSlots[queue.channel].ready.size() != queue.occupancy)
      return invalid("CGRA PE operand queue state diverged from its channel");
    if (!uniqueUnits.insert(queue.unitBinding).second)
      return invalid("CGRA PE operand activation repeats an allocation unit");
    units.push_back(queue.unitBinding);
  }

  for (std::uint64_t unitOrdinal : units) {
    const OperandQueueUnitBinding &unit = operandQueueUnits_[unitOrdinal];
    if (unit.occupancy > unit.capacity ||
        unit.reservations > unit.capacity - unit.occupancy)
      return invalid("CGRA PE operand allocation-unit occupancy is invalid");
    if (unit.occupancy + unit.reservations == unit.capacity)
      return false;
  }
  for (std::uint64_t unitOrdinal : units)
    ++operandQueueUnits_[unitOrdinal].reservations;
  inFlight.operandCapacityReserved = !units.empty();
  inFlight.operandCapacityBlocked = false;
  return true;
}

llvm::Error
CgraTransportRuntime::commitOperandQueueEnqueue(std::uint64_t slot) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA PE operand enqueue names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  llvm::SmallVector<std::uint64_t, 4> queues;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueUnits;
  for (const SinkBinding &sink :
       llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount)) {
    if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
      continue;
    if (sink.operandQueueBinding >= operandQueues_.size())
      return invalid("CGRA PE operand enqueue has an invalid queue binding");
    const OperandQueueBinding &queue =
        operandQueues_[sink.operandQueueBinding];
    if (queue.unitBinding >= operandQueueUnits_.size() ||
        queue.channel != sink.channel ||
        state_->channelSlots[queue.channel].ready.size() != queue.occupancy)
      return invalid("CGRA PE operand enqueue found divergent queue state");
    if (!uniqueUnits.insert(queue.unitBinding).second)
      return invalid("CGRA PE operand enqueue repeats an allocation unit");
    const OperandQueueUnitBinding &unit =
        operandQueueUnits_[queue.unitBinding];
    if (unit.reservations == 0 || unit.occupancy >= unit.capacity)
      return invalid("CGRA PE operand enqueue has no reserved capacity");
    queues.push_back(sink.operandQueueBinding);
  }
  if (queues.empty()) {
    if (inFlight.operandCapacityReserved)
      return invalid("CGRA non-queue transfer retained operand capacity");
    return llvm::Error::success();
  }
  if (!inFlight.operandCapacityReserved)
    return invalid("CGRA PE operand enqueue was not atomically reserved");

  for (std::uint64_t queueOrdinal : queues) {
    OperandQueueBinding &queue = operandQueues_[queueOrdinal];
    OperandQueueUnitBinding &unit = operandQueueUnits_[queue.unitBinding];
    --unit.reservations;
    ++unit.occupancy;
    ++queue.occupancy;
  }
  inFlight.operandCapacityReserved = false;
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::acceptActorCommits(
    llvm::ArrayRef<CgraActorLifecycleEvent> events) {
  struct Dequeue final {
    std::uint64_t queue = 0;
    std::uint64_t unit = 0;
  };
  llvm::SmallVector<Dequeue, 8> dequeues;
  llvm::SmallDenseSet<std::uint64_t, 8> frameUnits;
  for (const CgraActorLifecycleEvent &event : events) {
    if (event.kind != CgraActorLifecycleKind::Committed ||
        event.semanticActorOrdinal >= state_->execution->actorPlans.size())
      return invalid("CGRA PE operand dequeue has an invalid actor commit");
    const ActorExecutionPlan &actor =
        state_->execution->actorPlans[event.semanticActorOrdinal];
    const auto transition = llvm::find_if(
        actor.handshakeCases, [&](const auto &candidate) {
          return candidate.ordinal == event.transitionCaseOrdinal;
        });
    if (transition == actor.handshakeCases.end())
      return invalid("CGRA PE operand dequeue names an unknown transition");
    for (std::uint32_t input : transition->consumedInputs) {
      auto found = actorInputQueueBindings_.find(
          {event.semanticActorOrdinal, input});
      if (found == actorInputQueueBindings_.end())
        continue;
      if (found->second >= operandQueues_.size())
        return invalid("CGRA PE operand dequeue has an invalid queue binding");
      const OperandQueueBinding &queue = operandQueues_[found->second];
      if (queue.unitBinding >= operandQueueUnits_.size() ||
          queue.channel >= state_->channelSlots.size() ||
          queue.occupancy == 0 ||
          operandQueueUnits_[queue.unitBinding].occupancy == 0)
        return invalid("CGRA PE operand dequeue underflows its queue");
      const std::size_t channelOccupancy =
          state_->channelSlots[queue.channel].ready.size();
      if (channelOccupancy == std::numeric_limits<std::size_t>::max() ||
          channelOccupancy + 1 != queue.occupancy)
        return invalid("CGRA PE operand dequeue diverged from actor commit");
      if (!frameUnits.insert(queue.unitBinding).second)
        return invalid("CGRA PE operand dequeue service committed twice");
      dequeues.push_back({found->second, queue.unitBinding});
    }
  }
  for (const Dequeue &dequeue : dequeues) {
    --operandQueues_[dequeue.queue].occupancy;
    --operandQueueUnits_[dequeue.unit].occupancy;
  }
  return llvm::Error::success();
}

} // namespace loom::sim::detail

#include "CgraTransportRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <tuple>
#include <utility>
#include <variant>

// Diagnostic projections of the transport runtime's dynamic state. They add
// no identity the runtime does not already own: pending transfers, queue
// residency, exhausted virtual-channel rotations, and operand queue heads are
// read from the same bindings and queues the event execution mutates.

namespace loom::sim::detail {

const ::loom::mapping::SpatialPeOperandProgressFeedback &
CgraTransportRuntime::operandQueueProgress() const {
  return plan_->transport.operandQueueProgress;
}

std::vector<CgraPendingTransferDiagnostic>
CgraTransportRuntime::pendingTransferDiagnostics() const {
  std::vector<CgraPendingTransferDiagnostic> result;
  result.reserve(activeTransferCount_);
  const auto appendTraversalTargets =
      [&](const TraversalNodeBinding &node,
          std::vector<::loom::fabric::FabricPhysicalTraversalRef> &targets) {
        if (node.targetTraversalOffset > graph_.traversalTargets.size() ||
            node.targetTraversalCount >
                graph_.traversalTargets.size() - node.targetTraversalOffset)
          return;
        const auto selected =
            llvm::ArrayRef(graph_.traversalTargets)
                .slice(node.targetTraversalOffset, node.targetTraversalCount);
        targets.insert(targets.end(), selected.begin(), selected.end());
      };
  const auto storageHead = [&](std::uint64_t storageOrdinal)
      -> std::optional<CgraPendingTransferDiagnostic::StorageHead> {
    if (storageOrdinal >= storages_.size() ||
        storages_[storageOrdinal].queue.empty())
      return std::nullopt;
    const CgraTransportStorageEntry &head =
        storages_[storageOrdinal].queue.front();
    if (head.transferSlot >= inFlight_.size() ||
        !inFlight_[head.transferSlot].active)
      return std::nullopt;
    const InFlight &owner = inFlight_[head.transferSlot];
    return CgraPendingTransferDiagnostic::StorageHead{
        storageOrdinal, owner.bindingOrdinal, owner.occurrenceOrdinal,
        head.traversalNodeOrdinal};
  };
  for (auto [slot, transfer] : llvm::enumerate(inFlight_)) {
    if (!transfer.active)
      continue;
    const bool blocked = blocked_.test(slot);
    const bool operandCapacityReserved =
        llvm::any_of(transfer.publications,
                     [](const auto &state) { return state.capacityReserved; });
    const bool operandCapacityBlocked =
        llvm::any_of(transfer.publications,
                     [](const auto &state) { return state.capacityBlocked; });
    const std::uint32_t requestedPublicationCount =
        llvm::count_if(transfer.publications, [](const auto &state) {
          return state.consumedRequested;
        });
    const std::uint32_t publishedPublicationCount =
        llvm::count_if(transfer.publications,
                       [](const auto &state) { return state.published; });
    const std::uint32_t readySinkCount = llvm::count(transfer.readySinks, true);
    const std::uint32_t publishedSinkCount =
        llvm::count(transfer.publishedSinks, true);
    CgraPendingTransferDiagnostic diagnostic;
    diagnostic.bindingOrdinal = transfer.bindingOrdinal;
    diagnostic.occurrenceOrdinal = transfer.occurrenceOrdinal;
    diagnostic.blocked = blocked ||
                         (transfer.published &&
                          !transfer.producerCompletionReported);
    diagnostic.arrivalScheduled = transfer.arrivalScheduled;
    diagnostic.publicationReady = transfer.publicationReady;
    diagnostic.published = transfer.published;
    diagnostic.consumedRequested = transfer.consumedRequested;
    diagnostic.operandCapacityReserved = operandCapacityReserved;
    diagnostic.operandCapacityBlocked = operandCapacityBlocked;
    diagnostic.producedPermitted = transfer.producedPermitted;
    diagnostic.producedRetired = transfer.producedRetired;
    diagnostic.traversalPermitted = transfer.traversalPermitted;
    diagnostic.traversalRetired = transfer.traversalRetired;
    diagnostic.traversalTerminalsPermitted =
        transfer.traversalTerminalsPermitted;
    diagnostic.consumedPermitted = transfer.consumedPermitted;
    diagnostic.consumedRetired = transfer.consumedRetired;
    diagnostic.readySinkCount = readySinkCount;
    diagnostic.publishedSinkCount = publishedSinkCount;
    diagnostic.publicationCount =
        static_cast<std::uint32_t>(transfer.publications.size());
    diagnostic.requestedPublicationCount = requestedPublicationCount;
    diagnostic.publishedPublicationCount = publishedPublicationCount;
    if (transfer.bindingOrdinal < graph_.bindings.size()) {
      const TransferBinding &binding = graph_.bindings[transfer.bindingOrdinal];
      diagnostic.producer = binding.producer;
      diagnostic.sinkCount = binding.sinkCount;
      if (const auto *producer =
              std::get_if<::dataflow::ActorTokenResultRef>(&binding.producer)) {
        diagnostic.producerActorOrdinal =
            binding.semanticActorOrdinal.value_or(invalidCgraTransportOrdinal);
        diagnostic.producerResultOrdinal = producer->ordinal;
      }
      for (std::uint64_t node = binding.traversalNodeOffset;
           node != binding.traversalNodeOffset + binding.traversalNodeCount;
           ++node) {
        const std::uint64_t tagOrdinal =
            graph_.traversalNodes[node].physicalTagOrdinal;
        if (tagOrdinal == invalidCgraTransportOrdinal)
          continue;
        diagnostic.physicalTagOrdinal = tagOrdinal;
        if (tagOrdinal < plan_->transport.physicalTags.size()) {
          diagnostic.physicalTagValue =
              plan_->transport.physicalTags[tagOrdinal].value;
          diagnostic.physicalTagOwner =
              plan_->transport.physicalTags[tagOrdinal].mappingOwner;
        }
        break;
      }
      for (std::uint64_t node = binding.traversalNodeOffset;
           node != binding.traversalNodeOffset + binding.traversalNodeCount;
           ++node) {
        const TraversalNodeState state = traversalState(slot, node).state;
        if (state != TraversalNodeState::WaitingStorage &&
            state != TraversalNodeState::Queued)
          continue;
        const TraversalNodeBinding &traversal = graph_.traversalNodes[node];
        if (traversal.storageOrdinal >= storages_.size())
          continue;
        const StorageState &storage = storages_[traversal.storageOrdinal];
        diagnostic.blockingTraversalNodeOrdinal = node;
        diagnostic.blockingStorageOrdinal = traversal.storageOrdinal;
        appendTraversalTargets(traversal, diagnostic.blockingTraversals);
        for (std::uint64_t target = traversal.targetTraversalOffset;
             target !=
             traversal.targetTraversalOffset + traversal.targetTraversalCount;
             ++target) {
          if (target >= graph_.traversalTargets.size())
            continue;
          const auto *fifo =
              std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                  &graph_.traversalTargets[target].payload);
          if (!fifo ||
              fifo->mode != ::loom::fabric::FabricFifoTraversalMode::Buffered)
            continue;
          if (diagnostic.blockingFifoOccurrence &&
              *diagnostic.blockingFifoOccurrence != fifo->owner) {
            diagnostic.blockingFifoOccurrence.reset();
            break;
          }
          diagnostic.blockingFifoOccurrence = fifo->owner;
        }
        diagnostic.blockingStorageOccupancy = storage.queue.occupancy();
        diagnostic.blockingStorageReservations = storage.queue.reservations();
        diagnostic.blockingStorageCapacity = storage.queue.capacity();
        diagnostic.blockingStorageHead = storageHead(traversal.storageOrdinal);
        diagnostic.blockingTraversalWaitingForStorage =
            state == TraversalNodeState::WaitingStorage;
        diagnostic.blockingDownstreamStorageCount =
            static_cast<std::uint32_t>(traversal.downstreamStorageNodes.size());
        diagnostic.blockingUnbufferedSinkCount = static_cast<std::uint32_t>(
            traversal.unbufferedDescendantSinks.size());
        if (!traversal.downstreamStorageNodes.empty()) {
          const std::uint64_t downstream =
              traversal.downstreamStorageNodes.front();
          const TraversalNodeBinding &boundary =
              graph_.traversalNodes[downstream];
          appendTraversalTargets(boundary,
                                 diagnostic.blockingDownstreamTraversals);
          if (boundary.storageOrdinal < storages_.size()) {
            const StorageState &downstreamStorage =
                storages_[boundary.storageOrdinal];
            diagnostic.blockingDownstreamStorageOrdinal =
                boundary.storageOrdinal;
            diagnostic.blockingDownstreamStorageOccupancy =
                downstreamStorage.queue.occupancy();
            diagnostic.blockingDownstreamStorageReservations =
                downstreamStorage.queue.reservations();
            diagnostic.blockingDownstreamStorageCapacity =
                downstreamStorage.queue.capacity();
            diagnostic.blockingDownstreamStorageReserved =
                traversalState(slot, downstream).storageReserved;
            diagnostic.blockingDownstreamStorageHead =
                storageHead(boundary.storageOrdinal);
          }
        }
        break;
      }
      for (auto [localOrdinal, sink] :
           llvm::enumerate(llvm::ArrayRef(graph_.sinks)
                               .slice(binding.sinkOffset, binding.sinkCount))) {
        if (localOrdinal < transfer.publishedSinks.size() &&
            transfer.publishedSinks[localOrdinal]) {
          if (!transfer.acceptedSinks[localOrdinal] &&
              sink.kind == SinkKind::Channel &&
              diagnostic.blockingActorOrdinal == invalidCgraTransportOrdinal) {
            diagnostic.blockingActorOrdinal = sink.semanticActorOrdinal;
            diagnostic.blockingReadyTokenCount =
                state_->channelSlots[sink.channel].ready.size();
          }
          continue;
        }
        if (sink.kind != SinkKind::Channel) {
          diagnostic.unpublishedActorOrdinals.push_back(
              invalidCgraTransportOrdinal);
          diagnostic.unpublishedInputOrdinals.push_back(
              std::numeric_limits<std::uint32_t>::max());
          diagnostic.unpublishedReadyTokenCounts.push_back(0);
          continue;
        }
        const ChannelSlot &channel = state_->channelSlots[sink.channel];
        diagnostic.unpublishedActorOrdinals.push_back(
            sink.semanticActorOrdinal);
        diagnostic.unpublishedInputOrdinals.push_back(sink.inputOrdinal);
        diagnostic.unpublishedReadyTokenCounts.push_back(channel.ready.size());
        if (sink.operandQueueBinding == invalidCgraTransportOrdinal) {
          if (channel.ready.empty())
            continue;
          if (diagnostic.blockingActorOrdinal == invalidCgraTransportOrdinal) {
            diagnostic.blockingActorOrdinal = channel.ownerActorOrdinal;
            diagnostic.blockingReadyTokenCount = channel.ready.size();
          }
          continue;
        }
        if (sink.operandQueueBinding >= operandQueues_.size())
          continue;
        const OperandQueueState &queue =
            operandQueues_[sink.operandQueueBinding];
        if (queue.binding.unitBinding >= operandQueueUnits_.size())
          continue;
        const OperandQueueUnitState &unit =
            operandQueueUnits_[queue.binding.unitBinding];
        if (unit.occupancy <= unit.binding.capacity &&
            unit.reservations < unit.binding.capacity - unit.occupancy)
          continue;
        if (sink.operandActivationOrdinal >=
            plan_->transport.operandQueueActivations.size())
          continue;
        const auto &activation =
            plan_->transport
                .operandQueueActivations[sink.operandActivationOrdinal];
        diagnostic.operandQueueWaits.push_back(
            {queue.binding.queue, queue.binding.fu, activation.ingress,
             activation.tag, unit.binding.allocationUnit, unit.occupancy,
             unit.reservations, unit.binding.capacity});
        if (diagnostic.blockingActorOrdinal == invalidCgraTransportOrdinal) {
          diagnostic.blockingActorOrdinal = channel.ownerActorOrdinal;
          diagnostic.blockingQueueOccupancy = unit.occupancy;
          diagnostic.blockingQueueReservations = unit.reservations;
          diagnostic.blockingQueueCapacity = unit.binding.capacity;
        }
      }
    }
    result.push_back(std::move(diagnostic));
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.bindingOrdinal, lhs.occurrenceOrdinal) <
           std::tie(rhs.bindingOrdinal, rhs.occurrenceOrdinal);
  });
  return result;
}

std::vector<CgraStorageResidencyDiagnostic>
CgraTransportRuntime::storageResidencyDiagnostics(
    std::uint64_t storageOrdinal) const {
  std::vector<CgraStorageResidencyDiagnostic> residency;
  if (storageOrdinal >= storages_.size())
    return residency;
  std::vector<detail::CgraTransportStorageEntry> entries;
  storages_[storageOrdinal].queue.appendQueueOrder(entries);
  residency.reserve(entries.size());
  for (auto [position, entry] : llvm::enumerate(entries)) {
    CgraStorageResidencyDiagnostic record;
    record.queuePosition = static_cast<std::uint32_t>(position);
    record.traversalNodeOrdinal = entry.traversalNodeOrdinal;
    record.physicalTagOrdinal = entry.physicalTagOrdinal;
    record.virtualChannelKey = entry.virtualChannelKey;
    if (entry.physicalTagOrdinal < plan_->transport.physicalTags.size())
      record.physicalTagValue =
          plan_->transport.physicalTags[entry.physicalTagOrdinal].value;
    if (entry.transferSlot >= inFlight_.size() ||
        !inFlight_[entry.transferSlot].active) {
      residency.push_back(std::move(record));
      continue;
    }
    const InFlight &inFlight = inFlight_[entry.transferSlot];
    record.bindingOrdinal = inFlight.bindingOrdinal;
    record.occurrenceOrdinal = inFlight.occurrenceOrdinal;
    if (inFlight.bindingOrdinal < graph_.bindings.size() &&
        entry.traversalNodeOrdinal < graph_.traversalNodes.size()) {
      const TransferBinding &binding = graph_.bindings[inFlight.bindingOrdinal];
      record.producerActorOrdinal =
          binding.semanticActorOrdinal.value_or(invalidCgraTransportOrdinal);
      const TraversalNodeBinding &traversal =
          graph_.traversalNodes[entry.traversalNodeOrdinal];
      const auto &destinations =
          traversal.kind == TraversalNodeKind::BufferedStorage
              ? traversal.unbufferedDescendantSinks
              : traversal.descendantSinks;
      for (std::uint32_t localSink : destinations) {
        const std::uint64_t sink = binding.sinkOffset + localSink;
        if (sink >= graph_.sinks.size())
          break;
        const SinkBinding &binding = graph_.sinks[sink];
        if (binding.kind != SinkKind::Channel)
          continue;
        record.destinationChannelOrdinals.push_back(binding.channel);
        record.destinationActorOrdinals.push_back(binding.semanticActorOrdinal);
        record.destinationInputOrdinals.push_back(binding.inputOrdinal);
      }
    }
    residency.push_back(std::move(record));
  }
  return residency;
}

std::vector<CgraStorageOfferRotationDiagnostic>
CgraTransportRuntime::exhaustedOfferRotationDiagnostics() const {
  std::vector<CgraStorageOfferRotationDiagnostic> result;
  for (auto [storageOrdinal, storage] : llvm::enumerate(storages_)) {
    if (storage.binding.kind != CgraTraversalStorageKind::BufferedFifo ||
        storage.queue.discipline() !=
            ::fabric::FifoQueueDiscipline::PerTagVirtualChannel ||
        storage.queue.empty())
      continue;
    const std::uint32_t residentChannels =
        storage.queue.distinctResidentChannels();
    if (storage.offerRefusalsSinceCommit < residentChannels)
      continue;
    CgraStorageOfferRotationDiagnostic record;
    record.storageOrdinal = storageOrdinal;
    record.residentChannelCount = residentChannels;
    record.refusedOffersSinceCommit = storage.offerRefusalsSinceCommit;
    record.occupancy = storage.queue.occupancy();
    record.capacity = storage.queue.capacity();
    std::vector<CgraTransportStorageEntry> entries;
    storage.queue.appendQueueOrder(entries);
    // Every resident entry names the same selected FIFO occurrence through
    // its traversal node targets; a disagreement leaves the owner unnamed.
    bool ownerConflict = false;
    for (const CgraTransportStorageEntry &entry : entries) {
      if (entry.physicalTagOrdinal < plan_->transport.physicalTags.size()) {
        const llvm::APInt &value =
            plan_->transport.physicalTags[entry.physicalTagOrdinal].value;
        const bool seen =
            llvm::any_of(record.residentTagValues, [&](const llvm::APInt &tag) {
              return ::fabric::comparePhysicalTagValues(tag, value) == 0;
            });
        if (!seen)
          record.residentTagValues.push_back(value);
      }
      if (entry.traversalNodeOrdinal >= graph_.traversalNodes.size())
        continue;
      const TraversalNodeBinding &node =
          graph_.traversalNodes[entry.traversalNodeOrdinal];
      for (std::uint64_t target = node.targetTraversalOffset;
           target != node.targetTraversalOffset + node.targetTraversalCount;
           ++target) {
        if (target >= graph_.traversalTargets.size())
          continue;
        const auto *fifo =
            std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                &graph_.traversalTargets[target].payload);
        if (!fifo ||
            fifo->mode != ::loom::fabric::FabricFifoTraversalMode::Buffered)
          continue;
        if (record.fifoOccurrence && *record.fifoOccurrence != fifo->owner)
          ownerConflict = true;
        else
          record.fifoOccurrence = fifo->owner;
      }
    }
    if (ownerConflict)
      record.fifoOccurrence.reset();
    llvm::sort(record.residentTagValues,
               [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                 return ::fabric::comparePhysicalTagValues(lhs, rhs) < 0;
               });
    result.push_back(std::move(record));
  }
  return result;
}

std::vector<CgraOperandQueueHeadDiagnostic>
CgraTransportRuntime::pendingOperandQueueHeadDiagnostics() const {
  std::vector<CgraOperandQueueHeadDiagnostic> result;
  result.reserve(operandQueues_.size());
  for (const OperandQueueState &queue : operandQueues_) {
    if (queue.binding.unitBinding >= operandQueueUnits_.size())
      continue;
    const OperandQueueUnitState &unit =
        operandQueueUnits_[queue.binding.unitBinding];
    CgraOperandQueueHeadDiagnostic diagnostic{
        queue.binding.queue,
        queue.binding.fu,
        unit.binding.allocationUnit,
        unit.binding.capacity,
        queue.occupancy,
        unit.reservations,
        invalidCgraTransportOrdinal,
        invalidCgraTransportOrdinal,
        invalidCgraTransportOrdinal,
        llvm::APInt(1, 0),
        // An empty queue is an exact observation too: the absence of a head
        // must not be confused with an unknown queue representation.  A
        // non-empty queue is exact only when its retained entries account for
        // the complete occupancy.
        queue.entries.size() == queue.occupancy,
        {}};
    diagnostic.consumers.reserve(queue.binding.consumers.size());
    for (const auto &consumer : queue.binding.consumers)
      diagnostic.consumers.emplace_back(consumer.semanticActorOrdinal,
                                        consumer.inputOrdinal);
    if (!queue.entries.empty()) {
      const OperandQueueState::Entry &head = queue.entries.front();
      diagnostic.headBindingOrdinal = head.bindingOrdinal;
      diagnostic.headOccurrenceOrdinal = head.occurrenceOrdinal;
      diagnostic.headProducerSequenceOrdinal = head.producerSequenceOrdinal;
      diagnostic.headTag = head.tag;
      diagnostic.exactHead =
          diagnostic.exactHead &&
          head.bindingOrdinal != invalidCgraTransportOrdinal &&
          head.occurrenceOrdinal != invalidCgraTransportOrdinal &&
          head.producerSequenceOrdinal != invalidCgraTransportOrdinal;
    }
    result.push_back(std::move(diagnostic));
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return lhs.queue < rhs.queue;
  });
  return result;
}

} // namespace loom::sim::detail

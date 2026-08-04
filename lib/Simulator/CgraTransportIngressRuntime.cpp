#include "CgraTransportRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <limits>
#include <system_error>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

std::uint64_t CgraTransportRuntime::allocate(
    std::uint64_t bindingOrdinal, std::uint64_t occurrenceOrdinal,
    std::uint64_t producerSequenceOrdinal, Token token) {
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
  inFlight_[slot] = InFlight{true, bindingOrdinal, occurrenceOrdinal,
                             producerSequenceOrdinal, std::move(token)};
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
  case ActionStage::Storage:
    expectedClient = CgraPhysicalUseClientKind::TraversalTransport;
    break;
  case ActionStage::Consumed:
    expectedClient = CgraPhysicalUseClientKind::ConsumedTransport;
    break;
  }
  const auto appendAction = [&](std::uint64_t transferSlot,
                                std::uint64_t traversalNodeOrdinal,
                                std::uint64_t localActionOrdinal,
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
    ActionOwner owner;
    owner.transferSlot = transferSlot;
    owner.traversalNodeOrdinal = traversalNodeOrdinal;
    owner.stage = stage;
    owner.state = ActionLifecycleState::Requested;
    owner.localActionOrdinal = localActionOrdinal;
    owners.push_back(owner);
    increments[action] = increment + 1;
    return llvm::Error::success();
  };

  for (const PendingActionTransfer &transfer : transfers) {
    if (transfer.bindingOrdinal >= bindings_.size())
      return invalid("CGRA transport action names an unknown binding");
    const TransferBinding &binding = bindings_[transfer.bindingOrdinal];
    if (stage == ActionStage::Produced) {
      for (auto [localActionOrdinal, action] : llvm::enumerate(
               llvm::ArrayRef(physicalUses_)
                   .slice(binding.physicalUseOffset, binding.physicalUseCount)))
        if (llvm::Error error =
                appendAction(transfer.transferSlot, invalidCgraTransportOrdinal,
                             localActionOrdinal, action))
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
      const std::uint64_t localActionOrdinal = binding.physicalUseCount +
                                               transfer.traversalNodeOrdinal -
                                               binding.traversalNodeOffset;
      if (llvm::Error error =
              appendAction(transfer.transferSlot, transfer.traversalNodeOrdinal,
                           localActionOrdinal, action))
        return error;
      continue;
    }
    std::uint64_t localActionOrdinal =
        binding.physicalUseCount + binding.traversalNodeCount;
    for (const SinkBinding &sink :
         llvm::ArrayRef(sinks_).slice(binding.sinkOffset, binding.sinkCount))
      for (std::uint64_t action :
           llvm::ArrayRef(physicalUses_)
               .slice(sink.physicalUseOffset, sink.physicalUseCount))
        if (llvm::Error error =
                appendAction(transfer.transferSlot, invalidCgraTransportOrdinal,
                             localActionOrdinal++, action))
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
  llvm::SmallVector<std::uint64_t, 4> producerSequences;
  prospectiveSlots.reserve(transfers.size());
  producerSequences.reserve(transfers.size());
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
    if (binding.nextProducerSequenceOrdinal ==
        std::numeric_limits<std::uint64_t>::max())
      return llvm::createStringError(
          std::errc::value_too_large,
          "CGRA producer sequence ordinal overflows u64");
    if (std::holds_alternative<::dataflow::GraphIngressTokenRef>(
            binding.producer) &&
        transfer.occurrenceOrdinal != binding.nextProducerSequenceOrdinal)
      return invalid("CGRA graph-ingress producer sequence is not dense");
    producerSequences.push_back(binding.nextProducerSequenceOrdinal);
    producedTransfers.push_back({slot, transfer.bindingOrdinal});
  }

  auto requested =
      requestActions(producedTransfers, ActionStage::Produced, coordinate);
  if (!requested)
    return requested.takeError();

  llvm::SmallVector<std::uint64_t, 4> slots;
  slots.reserve(transfers.size());
  for (auto [transfer, expectedSlot, producerSequence] :
       llvm::zip(transfers, prospectiveSlots, producerSequences)) {
    const std::uint64_t slot =
        allocate(transfer.bindingOrdinal, transfer.occurrenceOrdinal,
                 producerSequence, std::move(*transfer.token));
    assert(slot == expectedSlot && "transport slot projection changed");
    ++bindings_[transfer.bindingOrdinal].nextProducerSequenceOrdinal;
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
    llvm::MutableArrayRef<CgraActorEmission> emissions) {
  if (emissions.empty())
    return llvm::Error::success();
  llvm::SmallVector<PendingTransfer, 4> transfers;
  llvm::SmallDenseSet<std::uint64_t, 4> uniqueBindings;
  transfers.reserve(emissions.size());
  for (CgraActorEmission &emission : emissions) {
    auto binding = actorSourceBindings_.find(
        {emission.semanticActorOrdinal, emission.resultOrdinal});
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

} // namespace loom::sim::detail

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

llvm::Expected<SpatialEventCoordinate>
nextPeClockBoundary(const SpatialEventCoordinate &coordinate) {
  const std::uint64_t numerator = coordinate.referenceCycle.numerator();
  const std::uint64_t denominator = coordinate.referenceCycle.denominator();
  const std::uint64_t cycle = numerator / denominator;
  if (cycle == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        std::errc::value_too_large,
        "CGRA PE operand retry coordinate overflows u64");
  auto boundary = ::loom::evaluation::ExactRatio::get(cycle + 1, 1);
  if (!boundary)
    return boundary.takeError();
  return SpatialEventCoordinate{*boundary, 0};
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
  TransferBinding &binding = bindings_[bindingOrdinal];
  InFlight transfer;
  transfer.active = true;
  transfer.bindingOrdinal = bindingOrdinal;
  transfer.occurrenceOrdinal = occurrenceOrdinal;
  transfer.producerSequenceOrdinal = producerSequenceOrdinal;
  transfer.token = std::move(token);
  transfer.publishedSinks.assign(binding.sinkCount, false);
  transfer.acceptedSinks.assign(binding.sinkCount, false);
  transfer.permittedSinkTerminals.assign(binding.sinkCount, 0);
  transfer.readySinks.assign(binding.sinkCount, false);
  transfer.publications.resize(binding.publicationCount);
  inFlight_[slot] = std::move(transfer);
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    traversalRemainingPredecessors_[nodeOrdinal] =
        traversalNodes_[nodeOrdinal].predecessorCount;
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Idle;
    traversalNodeTransferSlots_[nodeOrdinal] = slot;
  }
  binding.active = true;
  binding.sourceReserved = false;
  ++activeTransferCount_;
  return slot;
}

llvm::Expected<llvm::SmallVector<CgraPhysicalLifecycleEvent, 8>>
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
  const auto appendAction =
      [&](std::uint64_t transferSlot, std::uint64_t traversalNodeOrdinal,
          std::uint64_t publicationBinding, std::uint64_t localActionOrdinal,
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
    owner.publicationBinding = publicationBinding;
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
        if (llvm::Error error = appendAction(
                transfer.transferSlot, invalidCgraTransportOrdinal,
                invalidCgraTransportOrdinal, localActionOrdinal, action))
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
      if (llvm::Error error = appendAction(
              transfer.transferSlot, transfer.traversalNodeOrdinal,
              invalidCgraTransportOrdinal, localActionOrdinal, action))
        return error;
      continue;
    }
    if (transfer.publicationBinding < binding.publicationOffset ||
        transfer.publicationBinding >=
            binding.publicationOffset + binding.publicationCount)
      return invalid("CGRA consumed action names another publication");
    const PublicationBinding &publication =
        publications_[transfer.publicationBinding];
    for (std::uint32_t localSink :
         llvm::ArrayRef(publicationSinks_)
             .slice(publication.sinkOffset, publication.sinkCount)) {
      if (localSink >= binding.sinkCount)
        return invalid("CGRA publication names an unknown sink");
      const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
      std::uint64_t localActionOrdinal = binding.physicalUseCount +
                                         binding.traversalNodeCount +
                                         sink.consumedLocalActionOffset;
      for (std::uint64_t action :
           llvm::ArrayRef(physicalUses_)
               .slice(sink.physicalUseOffset, sink.physicalUseCount))
        if (llvm::Error error = appendAction(
                transfer.transferSlot, invalidCgraTransportOrdinal,
                transfer.publicationBinding, localActionOrdinal++, action))
          return error;
    }
  }

  if (requests.empty())
    return llvm::SmallVector<CgraPhysicalLifecycleEvent, 8>{};
  auto requested = physical_->requestBatch(requests, coordinate);
  if (!requested)
    return requested.takeError();

  for (const auto &[action, increment] : increments)
    nextActionOccurrence_[action] += increment;
  for (auto [request, owner] : llvm::zip(requests, owners)) {
    [[maybe_unused]] const bool inserted =
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
      auto directReady = markDirectSinksReady(slot);
      if (!directReady)
        return directReady.takeError();
      if (*directReady || binding.sinkCount == 0) {
        if (llvm::Error error = scheduleArrival(slot, *arrival))
          return error;
      }
      if (binding.traversalNodeCount != 0) {
        auto scheduled = scheduleReadyTraversals(slot, *arrival);
        if (!scheduled)
          return scheduled.takeError();
        if (!*scheduled)
          return invalid("CGRA traversal DAG has no ready root action");
      } else if (!inFlight_[slot].arrivalScheduled) {
        return invalid("CGRA direct transfer has no ready sink");
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
    if (bindings_[binding->second].active)
      return invalid(llvm::Twine("CGRA actor ") +
                     llvm::Twine(emission.semanticActorOrdinal) +
                     " occurrence " + llvm::Twine(emission.occurrenceOrdinal) +
                     " result " + llvm::Twine(emission.resultOrdinal) +
                     " reuses active transport binding " +
                     llvm::Twine(binding->second));
    if (!uniqueBindings.insert(binding->second).second)
      return invalid("CGRA actor emission batch repeats a source binding");
    if (emission.semanticActorOrdinal >= state_->execution->actorPlans.size())
      return invalid("CGRA actor emission names an unknown semantic actor");
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

llvm::Expected<bool>
CgraTransportRuntime::canAcceptGraphIngress(unsigned argumentOrdinal) const {
  auto binding = ingressSourceBindings_.find(argumentOrdinal);
  if (binding == ingressSourceBindings_.end())
    return invalid("CGRA graph ingress has no selected transfer binding");
  if (binding->second >= bindings_.size())
    return invalid("CGRA graph ingress binding exceeds the transport plan");
  return !bindings_[binding->second].active;
}

bool CgraTransportRuntime::actorSourcesAvailable(
    std::uint64_t semanticActorOrdinal) const {
  if (semanticActorOrdinal >= actorSourceBindingOrdinals_.size())
    return false;
  for (std::uint64_t binding :
       actorSourceBindingOrdinals_[semanticActorOrdinal])
    if (binding >= bindings_.size() || bindings_[binding].sourceReserved ||
        bindings_[binding].active)
      return false;
  return true;
}

llvm::Error
CgraTransportRuntime::retryBlocked(const SpatialEventCoordinate &coordinate) {
  auto publication = nextSpatialDelta(coordinate);
  if (!publication)
    return publication.takeError();
  auto operandAdmission = nextPeClockBoundary(coordinate);
  if (!operandAdmission)
    return operandAdmission.takeError();
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
    InFlight &inFlight = inFlight_[*slot];
    const TransferBinding &selected = bindings_[inFlight.bindingOrdinal];
    bool retryCapacity = false;
    bool retryPublication = false;
    for (std::uint32_t localPublication = 0;
         localPublication != selected.publicationCount; ++localPublication) {
      InFlight::PublicationState &state =
          inFlight.publications[localPublication];
      if (state.capacityBlocked) {
        state.capacityBlocked = false;
        retryCapacity = true;
      }
      retryPublication |=
          state.consumedRequested && !state.published &&
          state.consumedPermitted ==
              publications_[selected.publicationOffset + localPublication]
                  .consumedPhysicalUseCount;
    }
    if (retryPublication && !inFlight.publicationScheduled) {
      if (llvm::Error error = schedulePublication(*slot, *publication))
        return error;
    }
    if (retryCapacity && !inFlight.arrivalScheduled) {
      if (llvm::Error error = scheduleArrival(*slot, *operandAdmission))
        return error;
    }
    for (std::uint64_t node = selected.traversalNodeOffset;
         node != selected.traversalNodeOffset + selected.traversalNodeCount;
         ++node) {
      const TraversalNodeState state = traversalNodeStates_[node];
      if (state != TraversalNodeState::WaitingStorage &&
          state != TraversalNodeState::Queued)
        continue;
      const std::uint64_t storage = traversalNodes_[node].storageOrdinal;
      if (storage >= storages_.size())
        return invalid("CGRA blocked traversal has no storage owner");
      if (llvm::Error error = scheduleStorage(storage, *publication))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace loom::sim::detail

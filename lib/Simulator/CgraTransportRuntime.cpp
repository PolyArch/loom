#include "CgraTransportRuntime.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"

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

template <typename Key, typename Value>
Value &lookupOrAppend(
    llvm::SmallVectorImpl<std::pair<Key, Value>> &entries, const Key &key) {
  auto found = llvm::find_if(entries, [&](const auto &entry) {
    return entry.first == key;
  });
  if (found != entries.end())
    return found->second;
  entries.emplace_back(key, Value{});
  return entries.back().second;
}

} // namespace

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
  struct PublicationCountDelta final {
    std::uint32_t permitted = 0;
    std::uint32_t retired = 0;
  };
  using ActionKey = std::pair<std::uint64_t, std::uint64_t>;
  llvm::SmallVector<std::pair<ActionKey, ActionLifecycleState>, 8>
      projectedStates;
  llvm::SmallVector<std::pair<std::uint64_t, CountDelta>, 8> countDeltas;
  llvm::SmallVector<
      std::pair<std::pair<std::uint64_t, std::uint64_t>,
                PublicationCountDelta>,
      8>
      publicationDeltas;
  for (std::uint64_t storageOrdinal : touchedStorageFrameCommits_)
    storageFrameCommits_[storageOrdinal] = StorageFrameCommit{};
  touchedStorageFrameCommits_.clear();
  llvm::SmallVector<std::pair<std::uint64_t, std::uint64_t>, 8>
      newlyPermittedTraversals;
  const auto addTraversalPermission = [&](std::uint64_t slot,
                                          std::uint64_t node) -> llvm::Error {
    CountDelta &delta = lookupOrAppend(countDeltas, slot);
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

    auto projected = llvm::find_if(projectedStates, [&](const auto &entry) {
      return entry.first == key;
    });
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
    if (projected == projectedStates.end())
      projectedStates.emplace_back(key, state);
    else
      projected->second = state;
    CountDelta &delta = lookupOrAppend(countDeltas, owner.transferSlot);
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
        CountDelta &dequeueDelta = lookupOrAppend(countDeltas, dequeueSlot);
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
        CountDelta &enqueueDelta = lookupOrAppend(countDeltas, enqueueSlot);
        if (enqueueDelta.traversalRetired ==
            std::numeric_limits<std::uint32_t>::max())
          return invalid("CGRA storage retire count exceeds u32");
        ++enqueueDelta.traversalRetired;
      }
      continue;
    }
    case ActionStage::Consumed:
      if (owner.publicationBinding == invalidCgraTransportOrdinal)
        return invalid("CGRA consumed action has no publication owner");
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
    if (owner.stage == ActionStage::Consumed) {
      PublicationCountDelta &publication = lookupOrAppend(
          publicationDeltas,
          std::make_pair(owner.transferSlot, owner.publicationBinding));
      if ((permitted && publication.permitted ==
                            std::numeric_limits<std::uint32_t>::max()) ||
          (retired &&
           publication.retired == std::numeric_limits<std::uint32_t>::max()))
        return invalid("CGRA publication lifecycle count exceeds u32");
      publication.permitted += permitted;
      publication.retired += retired;
    }
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

  llvm::SmallVector<std::pair<std::uint64_t, std::uint32_t>, 8>
      successorDeltas;
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
      std::uint32_t &delta = lookupOrAppend(successorDeltas, successor);
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
        (delta.producedPermitted != 0 &&
         inFlight.producedPermitted + delta.producedPermitted ==
             binding.physicalUseCount) ||
        delta.traversalPermitted != 0 ||
        (!inFlight.publicationScheduled && !inFlight.published &&
         inFlight.consumedRequested && delta.consumedPermitted != 0 &&
         inFlight.consumedPermitted + delta.consumedPermitted ==
             binding.consumedPhysicalUseCount);
  }
  for (const auto &[key, delta] : publicationDeltas) {
    const auto [slot, publicationBinding] = key;
    if (slot >= inFlight_.size() || !inFlight_[slot].active)
      return invalid("CGRA publication lifecycle names an inactive token");
    const InFlight &inFlight = inFlight_[slot];
    const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    if (publicationBinding < binding.publicationOffset ||
        publicationBinding >=
            binding.publicationOffset + binding.publicationCount)
      return invalid("CGRA publication lifecycle names another transfer");
    const std::uint64_t localPublication =
        publicationBinding - binding.publicationOffset;
    const InFlight::PublicationState &state =
        inFlight.publications[localPublication];
    const PublicationBinding &publication = publications_[publicationBinding];
    if (!state.consumedRequested ||
        delta.permitted >
            publication.consumedPhysicalUseCount - state.consumedPermitted ||
        delta.retired >
            publication.consumedPhysicalUseCount - state.consumedRetired)
      return invalid("CGRA publication lifecycle exceeds selected uses");
    needsNextDelta |=
        delta.permitted != 0 && state.consumedPermitted + delta.permitted ==
                                    publication.consumedPhysicalUseCount;
  }
  std::optional<SpatialEventCoordinate> next;
  if (needsNextDelta) {
    auto coordinate = nextSpatialDelta(physicalFrame.coordinate);
    if (!coordinate)
      return coordinate.takeError();
    next = std::move(*coordinate);
  }

  llvm::SmallDenseSet<std::uint64_t, 4> storagesToSchedule;
  llvm::SmallDenseSet<std::uint64_t, 4> releasedStorageCapacity;
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
    if (commit.expectedDequeue && !commit.enqueue)
      releasedStorageCapacity.insert(storageOrdinal);
    if (commit.enqueue) {
      auto pending =
          llvm::find(storage.pendingEnqueueNodes, commit.enqueueNode);
      if (pending == storage.pendingEnqueueNodes.end())
        return invalid("CGRA storage commit lost its enqueue request");
      storage.pendingEnqueueNodes.erase(pending);
      if (traversalStorageReserved_[commit.enqueueNode]) {
        if (storage.reservations == 0)
          return invalid("CGRA downstream storage reservation underflow");
        --storage.reservations;
        traversalStorageReserved_[commit.enqueueNode] = false;
      }
      if (storage.kind == CgraTraversalStorageKind::BufferedFifo)
        traversalNodeStates_[commit.enqueueNode] = TraversalNodeState::Queued;
      else
        ++inFlight_[commit.enqueue->transferSlot].traversalPermitted;
      const TraversalNodeBinding &enqueueTraversal =
          traversalNodes_[commit.enqueueNode];
      if (enqueueTraversal.kind == TraversalNodeKind::BufferedStorage ||
          enqueueTraversal.kind == TraversalNodeKind::RegisterStorageWrite) {
        if (llvm::Error error = acceptDurableSinks(
                commit.enqueue->transferSlot, enqueueTraversal.descendantSinks))
          return std::move(error);
        auto producerCompletion =
            maybeCompleteProducer(commit.enqueue->transferSlot);
        if (!producerCompletion)
          return producerCompletion.takeError();
        if (*producerCompletion)
          completions.push_back(**producerCompletion);
      }
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
    std::uint32_t *publicationPermittedCount = nullptr;
    std::uint32_t *publicationRetiredCount = nullptr;
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
      if (owner.publicationBinding <
              bindings_[inFlight.bindingOrdinal].publicationOffset ||
          owner.publicationBinding >=
              bindings_[inFlight.bindingOrdinal].publicationOffset +
                  bindings_[inFlight.bindingOrdinal].publicationCount)
        return invalid("CGRA consumed action names another publication");
      {
        InFlight::PublicationState &publication =
            inFlight.publications[owner.publicationBinding -
                                  bindings_[inFlight.bindingOrdinal]
                                      .publicationOffset];
        publicationPermittedCount = &publication.consumedPermitted;
        publicationRetiredCount = &publication.consumedRetired;
      }
      break;
    }
    switch (event.kind) {
    case CgraPhysicalLifecycleKind::Requested:
      llvm_unreachable("request lifecycle rejected above");
    case CgraPhysicalLifecycleKind::Granted:
      owner.state = requiresCommit ? ActionLifecycleState::Granted
                                   : ActionLifecycleState::Permitted;
      if (!requiresCommit) {
        ++*permittedCount;
        if (publicationPermittedCount)
          ++*publicationPermittedCount;
      }
      break;
    case CgraPhysicalLifecycleKind::Committed:
      owner.state = ActionLifecycleState::Permitted;
      ++*permittedCount;
      if (publicationPermittedCount)
        ++*publicationPermittedCount;
      break;
    case CgraPhysicalLifecycleKind::Retired:
      ++*retiredCount;
      if (publicationRetiredCount)
        ++*publicationRetiredCount;
      actionOwners_.erase(indexed);
      break;
    }
  }

  llvm::SmallDenseSet<std::uint64_t, 4> newlyReadySlots;
  for (const auto &[slot, nodeOrdinal] : newlyPermittedTraversals) {
    InFlight &inFlight = inFlight_[slot];
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Permitted;
    if (traversalNodes_[nodeOrdinal].terminal)
      ++inFlight.traversalTerminalsPermitted;
    auto ready = markTerminalSinksReady(slot, nodeOrdinal);
    if (!ready)
      return ready.takeError();
    if (*ready)
      newlyReadySlots.insert(slot);
  }
  for (const auto &[successor, delta] : successorDeltas)
    traversalRemainingPredecessors_[successor] -= delta;

  for (const auto &[slot, delta] : countDeltas) {
    InFlight &inFlight = inFlight_[slot];
    const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    if (delta.producedPermitted != 0 &&
        inFlight.producedPermitted == binding.physicalUseCount) {
      if (!next)
        return invalid("CGRA traversal request lost its next delta");
      auto directReady = markDirectSinksReady(slot);
      if (!directReady)
        return directReady.takeError();
      if (*directReady || binding.sinkCount == 0)
        newlyReadySlots.insert(slot);
      if (binding.traversalNodeCount != 0) {
        auto scheduled = scheduleReadyTraversals(slot, *next);
        if (!scheduled)
          return scheduled.takeError();
        if (!*scheduled)
          return invalid("CGRA traversal DAG has no ready root action");
      }
    }
    if (delta.traversalPermitted != 0) {
      if (!next)
        return invalid("CGRA traversal successor lost its next delta");
      auto scheduled = scheduleReadyTraversals(slot, *next);
      if (!scheduled)
        return scheduled.takeError();
    }
    if (newlyReadySlots.contains(slot) && !inFlight.arrivalScheduled) {
      if (!next)
        return invalid("CGRA sink arrival lost its next delta");
      if (llvm::Error error = scheduleArrival(slot, *next))
        return error;
    }
    if (auto completion = maybeRelease(slot))
      completions.push_back(*completion);
  }
  llvm::SmallDenseSet<std::uint64_t, 4> publicationReadySlots;
  for (const auto &[key, delta] : publicationDeltas) {
    const auto [slot, publicationBinding] = key;
    const InFlight &inFlight = inFlight_[slot];
    const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
    const InFlight::PublicationState &state =
        inFlight.publications[publicationBinding - binding.publicationOffset];
    if (delta.permitted != 0 &&
        state.consumedPermitted ==
            publications_[publicationBinding].consumedPhysicalUseCount)
      publicationReadySlots.insert(slot);
  }
  for (std::uint64_t slot : publicationReadySlots) {
    if (!next)
      return invalid("CGRA publication lost its next delta");
    if (!inFlight_[slot].publicationScheduled && !inFlight_[slot].published)
      if (llvm::Error error = schedulePublication(slot, *next))
        return error;
  }
  for (std::uint64_t storageOrdinal : releasedStorageCapacity)
    for (std::uint64_t upstream :
         storages_[storageOrdinal].upstreamStorageOrdinals)
      storagesToSchedule.insert(upstream);
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

bool CgraTransportRuntime::canPublishSink(const SinkBinding &sink,
                                          bool operandCapacityReserved) const {
  if (sink.kind != SinkKind::Channel)
    return true;
  if (sink.channel >= state_->channelSlots.size())
    return false;
  if (sink.operandQueueBinding == invalidCgraTransportOrdinal)
    return state_->channelSlots[sink.channel].ready.empty();
  if (sink.operandQueueBinding >= operandQueues_.size())
    return false;
  const OperandQueueBinding &queue = operandQueues_[sink.operandQueueBinding];
  if (queue.unitBinding >= operandQueueUnits_.size())
    return false;
  const OperandQueueUnitBinding &unit = operandQueueUnits_[queue.unitBinding];
  return operandCapacityReserved ||
         (unit.occupancy <= unit.capacity &&
          unit.reservations < unit.capacity - unit.occupancy);
}

bool CgraTransportRuntime::canPublish(std::uint64_t slot,
                                      std::uint64_t publicationBinding) const {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return false;
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return false;
  const PublicationBinding &publication = publications_[publicationBinding];
  const InFlight::PublicationState &publicationState =
      inFlight.publications[publicationBinding - binding.publicationOffset];
  for (std::uint32_t localSink :
       llvm::ArrayRef(publicationSinks_)
           .slice(publication.sinkOffset, publication.sinkCount)) {
    if (localSink >= binding.sinkCount || !inFlight.readySinks[localSink] ||
        inFlight.publishedSinks[localSink])
      return false;
    const SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
    if (!canPublishSink(sink, publicationState.capacityReserved ||
                                  publicationState.enqueueCommitted))
      return false;
  }
  return true;
}

bool CgraTransportRuntime::canPublishSinks(
    const TransferBinding &binding, bool operandCapacityReserved,
    llvm::ArrayRef<std::uint32_t> localSinkOrdinals) const {
  for (std::uint32_t ordinal : localSinkOrdinals) {
    if (ordinal >= binding.sinkCount)
      return false;
    const SinkBinding &sink = sinks_[binding.sinkOffset + ordinal];
    if (!canPublishSink(sink, operandCapacityReserved))
      return false;
  }
  return true;
}

bool CgraTransportRuntime::canAdvanceBufferedStorage(
    std::uint64_t slot, std::uint64_t nodeOrdinal) const {
  if (slot >= inFlight_.size() || !inFlight_[slot].active ||
      nodeOrdinal >= traversalNodes_.size())
    return false;
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  const TraversalNodeBinding &node = traversalNodes_[nodeOrdinal];
  if (node.kind != TraversalNodeKind::BufferedStorage ||
      !canPublishSinks(binding, /*operandCapacityReserved=*/false,
                       node.unbufferedDescendantSinks))
    return false;
  for (std::uint64_t downstream : node.downstreamStorageNodes) {
    if (downstream >= traversalNodes_.size() ||
        traversalStorageReserved_[downstream])
      return false;
    const TraversalNodeBinding &boundary = traversalNodes_[downstream];
    if (boundary.kind == TraversalNodeKind::PhysicalAction ||
        boundary.storageOrdinal >= storages_.size())
      return false;
    const StorageBinding &storage = storages_[boundary.storageOrdinal];
    if (storage.queue.occupancy() > storage.queue.capacity() ||
        storage.reservations >=
            storage.queue.capacity() - storage.queue.occupancy())
      return false;
  }
  return true;
}

llvm::Error
CgraTransportRuntime::reserveDownstreamStorage(std::uint64_t slot,
                                               std::uint64_t nodeOrdinal) {
  if (!canAdvanceBufferedStorage(slot, nodeOrdinal))
    return invalid("CGRA buffered dequeue lost downstream capacity");
  for (std::uint64_t downstream :
       traversalNodes_[nodeOrdinal].downstreamStorageNodes) {
    TraversalNodeBinding &boundary = traversalNodes_[downstream];
    StorageBinding &storage = storages_[boundary.storageOrdinal];
    if (storage.reservations == std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA downstream storage reservation exceeds u32");
    ++storage.reservations;
    traversalStorageReserved_[downstream] = true;
  }
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::acceptDurableSinks(
    std::uint64_t slot, llvm::ArrayRef<std::uint32_t> localSinks) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA durable acceptance names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (inFlight.acceptedSinks.size() != binding.sinkCount)
    return invalid("CGRA durable acceptance has the wrong sink domain");
  for (std::uint32_t sink : localSinks) {
    if (sink >= binding.sinkCount)
      return invalid("CGRA durable acceptance names an unknown sink");
    if (inFlight.acceptedSinks[sink])
      continue;
    inFlight.acceptedSinks[sink] = true;
    ++inFlight.acceptedSinkCount;
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<CgraTransportCompletion>>
CgraTransportRuntime::maybeCompleteProducer(std::uint64_t slot) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA producer completion names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (inFlight.acceptedSinkCount > binding.sinkCount)
    return invalid("CGRA producer acceptance exceeds its sink domain");
  if (inFlight.producerCompletionReported ||
      inFlight.acceptedSinkCount != binding.sinkCount)
    return std::optional<CgraTransportCompletion>();
  inFlight.producerCompletionReported = true;
  if (!binding.semanticActorOrdinal)
    return std::optional<CgraTransportCompletion>();
  return std::optional<CgraTransportCompletion>(CgraTransportCompletion{
      *binding.semanticActorOrdinal, inFlight.occurrenceOrdinal});
}

llvm::Error CgraTransportRuntime::publish(std::uint64_t slot,
                                          std::uint64_t publicationBinding,
                                          CgraTransportFrame &frame) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA publication names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (publicationBinding < binding.publicationOffset ||
      publicationBinding >=
          binding.publicationOffset + binding.publicationCount)
    return invalid("CGRA publication names another transfer");
  const std::uint64_t localPublication =
      publicationBinding - binding.publicationOffset;
  InFlight::PublicationState &publicationState =
      inFlight.publications[localPublication];
  if (publicationState.published || !publicationState.consumedRequested ||
      publicationState.consumedPermitted !=
          publications_[publicationBinding].consumedPhysicalUseCount ||
      !canPublish(slot, publicationBinding))
    return invalid("CGRA publication instance is not ready");
  if (!publicationState.enqueueCommitted)
    return invalid("CGRA publication omitted its operand enqueue commit");
  const auto publicationSinkRange =
      llvm::ArrayRef(publicationSinks_)
          .slice(publications_[publicationBinding].sinkOffset,
                 publications_[publicationBinding].sinkCount);
  for (std::uint32_t localSink : publicationSinkRange) {
    SinkBinding &sink = sinks_[binding.sinkOffset + localSink];
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
    inFlight.publishedSinks[localSink] = true;
    ++inFlight.publishedSinkCount;
  }
  if (llvm::Error error = acceptDurableSinks(slot, publicationSinkRange))
    return error;
  auto producerCompletion = maybeCompleteProducer(slot);
  if (!producerCompletion)
    return producerCompletion.takeError();
  if (*producerCompletion)
    frame.completions.push_back(**producerCompletion);
  publicationState.published = true;
  inFlight.publicationReady = false;
  if (llvm::any_of(inFlight.publications,
                   [](const auto &state) { return !state.published; }))
    return llvm::Error::success();
  if (inFlight.publishedSinkCount != binding.sinkCount)
    return invalid("CGRA publication coverage disagrees with its sink domain");
  if (!binding.discard)
    frame.publications.push_back({binding.producer, inFlight.occurrenceOrdinal,
                                  inFlight.producerSequenceOrdinal,
                                  std::move(inFlight.token)});
  inFlight.published = true;
  if (auto completion = maybeRelease(slot))
    frame.completions.push_back(*completion);
  return llvm::Error::success();
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
  if (binding.semanticActorOrdinal && !inFlight.producerCompletionReported) {
    inFlight.producerCompletionReported = true;
    completion = CgraTransportCompletion{*binding.semanticActorOrdinal,
                                         inFlight.occurrenceOrdinal};
  }
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    assert(!traversalStorageReserved_[nodeOrdinal] &&
           "retired CGRA transfer retained storage capacity");
    traversalRemainingPredecessors_[nodeOrdinal] = 0;
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Idle;
    traversalNodeTransferSlots_[nodeOrdinal] = invalidCgraTransportOrdinal;
  }
  binding.active = false;
  binding.sourceReserved = false;
  if (binding.semanticActorOrdinal &&
      actorSourcesAvailable(*binding.semanticActorOrdinal))
    state_->nextActorCandidates.set(*binding.semanticActorOrdinal);
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
    auto requested = requestedEvents_.popNextFrameView();
    if (!requested)
      return requested.takeError();
    for (const CgraScheduledEvent &event : (**requested).events)
      frame.physicalEvents.push_back(
          {CgraPhysicalLifecycleKind::Requested,
           event.order.structuralActionOrdinal, event.order.occurrenceOrdinal,
           event.order.ownerEventOrdinal, event.order.coordinate});
  }

  if (isAt(traversalEvents_.nextCoordinate(), *coordinate)) {
    auto traversals = traversalEvents_.popNextFrameView();
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
    auto storageFrame = storageEvents_.popNextFrameView();
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
        const auto reserved = llvm::find_if(
            storage.pendingEnqueueNodes, [&](std::uint64_t candidate) {
              return candidate < traversalStorageReserved_.size() &&
                     traversalStorageReserved_[candidate];
            });
        const std::uint64_t candidate =
            reserved == storage.pendingEnqueueNodes.end()
                ? storage.pendingEnqueueNodes.front()
                : *reserved;
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
        const TraversalNodeBinding &headNode =
            traversalNodes_[head.traversalNodeOrdinal];
        if (headNode.descendantSinks.empty())
          return invalid("CGRA storage traversal reaches no logical sink");
        if (storage.kind == CgraTraversalStorageKind::BufferedFifo &&
            canAdvanceBufferedStorage(head.transferSlot,
                                      head.traversalNodeOrdinal)) {
          dequeueEntry = head;
          dequeueNode = head.traversalNodeOrdinal;
        } else if (storage.kind != CgraTraversalStorageKind::BufferedFifo &&
                   canPublishSinks(binding,
                                   /*operandCapacityReserved=*/false,
                                   headNode.descendantSinks)) {
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
      const bool enqueueReserved =
          enqueueNode && traversalStorageReserved_[*enqueueNode];
      const bool unreservedCapacity =
          storage.queue.occupancy() <= storage.queue.capacity() &&
          storage.reservations <
              storage.queue.capacity() - storage.queue.occupancy();
      bool enqueue =
          enqueueNode.has_value() && (enqueueReserved || unreservedCapacity);
      if (enqueueNode && dequeue &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo)
        enqueue = storage.independentReadWriteServices;
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
      const bool independentReplacement =
          enqueue && dequeue &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo &&
          storage.independentReadWriteServices;
      if (enqueue && !independentReplacement &&
          !traversalStorageReserved_[*enqueueNode]) {
        if (storage.queue.occupancy() > storage.queue.capacity() ||
            storage.reservations >=
                storage.queue.capacity() - storage.queue.occupancy())
          return invalid(llvm::Twine("CGRA storage enqueue node ") +
                         llvm::Twine(*enqueueNode) + " at storage " +
                         llvm::Twine(storageOrdinal) + " has " +
                         llvm::Twine(storage.queue.occupancy()) + "+" +
                         llvm::Twine(storage.reservations) + "/" +
                         llvm::Twine(storage.queue.capacity()));
        ++storage.reservations;
        traversalStorageReserved_[*enqueueNode] = true;
      }
      if (storage.kind == CgraTraversalStorageKind::BufferedFifo &&
          dequeueEntry)
        if (llvm::Error error = reserveDownstreamStorage(
                dequeueEntry->transferSlot, dequeueEntry->traversalNodeOrdinal))
          return std::move(error);
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
    auto arrivals = arrivalEvents_.popNextFrameView();
    if (!arrivals)
      return arrivals.takeError();
    llvm::SmallVector<PendingActionTransfer, 4> transfers;
    llvm::SmallVector<std::pair<std::uint64_t, std::uint64_t>, 4>
        requestedPublications;
    llvm::SmallDenseSet<std::uint64_t, 4> publicationSlots;
    struct ReadyPublication final {
      std::uint64_t slot = 0;
      std::uint64_t publication = 0;
      OperandIngressAdmission admission;
    };
    llvm::SmallVector<ReadyPublication, 8> readyPublications;
    transfers.reserve((**arrivals).events.size());
    requestedPublications.reserve((**arrivals).events.size());
    for (const CgraScheduledEvent &event : (**arrivals).events) {
      if (event.payload >= inFlight_.size() || !inFlight_[event.payload].active)
        return invalid("CGRA transport arrival names an inactive token");
      InFlight &inFlight = inFlight_[event.payload];
      if (!inFlight.arrivalScheduled ||
          inFlight.bindingOrdinal != event.order.structuralActionOrdinal ||
          inFlight.occurrenceOrdinal != event.order.occurrenceOrdinal)
        return invalid("CGRA transport arrival key is inconsistent");
      inFlight.arrivalScheduled = false;
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      if (inFlight.readySinks.size() != binding.sinkCount ||
          inFlight.publishedSinks.size() != binding.sinkCount ||
          inFlight.publications.size() != binding.publicationCount)
        return invalid("CGRA sink arrival state has the wrong domain");
      for (std::uint32_t localPublication = 0;
           localPublication != binding.publicationCount; ++localPublication) {
        InFlight::PublicationState &state =
            inFlight.publications[localPublication];
        if (state.consumedRequested || state.published)
          continue;
        const std::uint64_t publicationBinding =
            binding.publicationOffset + localPublication;
        const PublicationBinding &publication =
            publications_[publicationBinding];
        bool ready = true;
        for (std::uint32_t localSink :
             llvm::ArrayRef(publicationSinks_)
                 .slice(publication.sinkOffset, publication.sinkCount)) {
          if (localSink >= binding.sinkCount)
            return invalid("CGRA publication arrival names an unknown sink");
          ready &= inFlight.readySinks[localSink];
        }
        if (!ready)
          continue;
        auto priority =
            operandIngressAdmissionPriority(event.payload, publicationBinding);
        if (!priority)
          return priority.takeError();
        readyPublications.push_back(
            {event.payload, publicationBinding, std::move(*priority)});
      }
    }
    llvm::stable_sort(readyPublications, [](const auto &lhs, const auto &rhs) {
      return static_cast<std::uint8_t>(lhs.admission.priority) >
             static_cast<std::uint8_t>(rhs.admission.priority);
    });
    llvm::SmallVector<const ReadyPublication *, 8> observedPriority;
    for (const ReadyPublication &candidate : readyPublications) {
      InFlight &inFlight = inFlight_[candidate.slot];
      const PublicationBinding &publication =
          publications_[candidate.publication];
      const bool suppressed =
          llvm::any_of(observedPriority, [&](const ReadyPublication *selected) {
            if (static_cast<std::uint8_t>(selected->admission.priority) <=
                static_cast<std::uint8_t>(candidate.admission.priority))
              return false;
            return llvm::any_of(candidate.admission.pairings,
                                [&](const auto &pairing) {
                                  return llvm::is_contained(
                                      selected->admission.pairings, pairing);
                                });
          });
      observedPriority.push_back(&candidate);
      if (suppressed) {
        const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
        InFlight::PublicationState &state =
            inFlight.publications[candidate.publication -
                                  binding.publicationOffset];
        state.capacityBlocked = true;
        blocked_.set(inFlight.bindingOrdinal);
        frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
        continue;
      }
      auto capacityReady = reserveOperandQueueCapacity(
          candidate.slot, candidate.publication, (**arrivals).coordinate);
      if (!capacityReady)
        return capacityReady.takeError();
      if (!*capacityReady) {
        blocked_.set(inFlight.bindingOrdinal);
        frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
        continue;
      }
      transfers.push_back({candidate.slot, inFlight.bindingOrdinal,
                           invalidCgraTransportOrdinal, candidate.publication});
      requestedPublications.emplace_back(candidate.slot, candidate.publication);
      if (publication.consumedPhysicalUseCount == 0)
        publicationSlots.insert(candidate.slot);
    }
    auto requested = requestActions(transfers, ActionStage::Consumed,
                                    (**arrivals).coordinate);
    if (!requested)
      return requested.takeError();
    frame.physicalEvents.insert(frame.physicalEvents.end(), requested->begin(),
                                requested->end());
    for (const auto &[slot, publicationBinding] : requestedPublications) {
      InFlight &inFlight = inFlight_[slot];
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      inFlight.publications[publicationBinding - binding.publicationOffset]
          .consumedRequested = true;
      inFlight.consumedRequested = true;
    }
    for (std::uint64_t slot : publicationSlots)
      if (!inFlight_[slot].publicationScheduled && !inFlight_[slot].published)
        if (llvm::Error error = schedulePublication(slot, *coordinate))
          return error;
  }

  if (isAt(events_.nextCoordinate(), *coordinate)) {
    auto publications = events_.popNextFrameView();
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
      inFlight.publicationReady = true;
      const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
      bool readyPublicationBlocked = false;
      for (std::uint32_t localPublication = 0;
           localPublication != binding.publicationCount; ++localPublication) {
        const std::uint64_t publicationBinding =
            binding.publicationOffset + localPublication;
        const PublicationBinding &publication =
            publications_[publicationBinding];
        InFlight::PublicationState &state =
            inFlight.publications[localPublication];
        if (state.published || !state.consumedRequested ||
            state.consumedPermitted != publication.consumedPhysicalUseCount)
          continue;
        if (!canPublish(event.payload, publicationBinding)) {
          readyPublicationBlocked = true;
          continue;
        }
        if (!state.enqueueCommitted) {
          if (llvm::Error error =
                  commitOperandQueueEnqueue(event.payload, publicationBinding))
            return std::move(error);
          state.enqueueCommitted = true;
        }
        if (llvm::Error error =
                publish(event.payload, publicationBinding, frame))
          return std::move(error);
      }
      inFlight.publicationReady = false;
      if (!inFlight.published && readyPublicationBlocked) {
        if (!blocked_.test(inFlight.bindingOrdinal))
          frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
        blocked_.set(inFlight.bindingOrdinal);
      }
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

std::vector<CgraPendingTransferDiagnostic>
CgraTransportRuntime::pendingTransferDiagnostics() const {
  std::vector<CgraPendingTransferDiagnostic> result;
  result.reserve(activeTransferCount_);
  const auto appendTraversalTargets =
      [&](const TraversalNodeBinding &node,
          std::vector<::loom::fabric::FabricPhysicalTraversalRef> &targets) {
        if (node.targetTraversalOffset > traversalTargets_.size() ||
            node.targetTraversalCount >
                traversalTargets_.size() - node.targetTraversalOffset)
          return;
        const auto selected =
            llvm::ArrayRef(traversalTargets_)
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
  for (const InFlight &transfer : inFlight_) {
    if (!transfer.active)
      continue;
    const bool blocked = transfer.bindingOrdinal < blocked_.size() &&
                         blocked_.test(transfer.bindingOrdinal);
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
    diagnostic.blocked = blocked;
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
    if (transfer.bindingOrdinal < bindings_.size()) {
      const TransferBinding &binding = bindings_[transfer.bindingOrdinal];
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
        const TraversalNodeState state = traversalNodeStates_[node];
        if (state != TraversalNodeState::WaitingStorage &&
            state != TraversalNodeState::Queued)
          continue;
        const TraversalNodeBinding &traversal = traversalNodes_[node];
        if (traversal.storageOrdinal >= storages_.size())
          continue;
        const StorageBinding &storage = storages_[traversal.storageOrdinal];
        diagnostic.blockingTraversalNodeOrdinal = node;
        diagnostic.blockingStorageOrdinal = traversal.storageOrdinal;
        appendTraversalTargets(traversal, diagnostic.blockingTraversals);
        for (std::uint64_t target = traversal.targetTraversalOffset;
             target !=
             traversal.targetTraversalOffset + traversal.targetTraversalCount;
             ++target) {
          if (target >= traversalTargets_.size())
            continue;
          const auto *fifo =
              std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                  &traversalTargets_[target].payload);
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
        diagnostic.blockingStorageReservations = storage.reservations;
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
          const TraversalNodeBinding &boundary = traversalNodes_[downstream];
          appendTraversalTargets(boundary,
                                 diagnostic.blockingDownstreamTraversals);
          if (boundary.storageOrdinal < storages_.size()) {
            const StorageBinding &downstreamStorage =
                storages_[boundary.storageOrdinal];
            diagnostic.blockingDownstreamStorageOrdinal =
                boundary.storageOrdinal;
            diagnostic.blockingDownstreamStorageOccupancy =
                downstreamStorage.queue.occupancy();
            diagnostic.blockingDownstreamStorageReservations =
                downstreamStorage.reservations;
            diagnostic.blockingDownstreamStorageCapacity =
                downstreamStorage.queue.capacity();
            diagnostic.blockingDownstreamStorageReserved =
                traversalStorageReserved_[downstream];
            diagnostic.blockingDownstreamStorageHead =
                storageHead(boundary.storageOrdinal);
          }
        }
        break;
      }
      for (auto [localOrdinal, sink] :
           llvm::enumerate(llvm::ArrayRef(sinks_).slice(binding.sinkOffset,
                                                        binding.sinkCount))) {
        if (localOrdinal < transfer.publishedSinks.size() &&
            transfer.publishedSinks[localOrdinal])
          continue;
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
        const OperandQueueBinding &queue =
            operandQueues_[sink.operandQueueBinding];
        if (queue.unitBinding >= operandQueueUnits_.size())
          continue;
        const OperandQueueUnitBinding &unit =
            operandQueueUnits_[queue.unitBinding];
        if (unit.occupancy <= unit.capacity &&
            unit.reservations < unit.capacity - unit.occupancy)
          continue;
        if (sink.operandActivationOrdinal >=
            plan_->transport.operandQueueActivations.size())
          continue;
        const auto &activation =
            plan_->transport
                .operandQueueActivations[sink.operandActivationOrdinal];
        diagnostic.operandQueueWaits.push_back(
            {queue.queue, queue.fu, activation.ingress, activation.tag,
             unit.allocationUnit, unit.occupancy, unit.reservations,
             unit.capacity});
        if (diagnostic.blockingActorOrdinal == invalidCgraTransportOrdinal) {
          diagnostic.blockingActorOrdinal = channel.ownerActorOrdinal;
          diagnostic.blockingQueueOccupancy = unit.occupancy;
          diagnostic.blockingQueueReservations = unit.reservations;
          diagnostic.blockingQueueCapacity = unit.capacity;
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

std::vector<CgraOperandQueueHeadDiagnostic>
CgraTransportRuntime::pendingOperandQueueHeadDiagnostics() const {
  std::vector<CgraOperandQueueHeadDiagnostic> result;
  result.reserve(operandQueues_.size());
  for (const OperandQueueBinding &queue : operandQueues_) {
    if (queue.unitBinding >= operandQueueUnits_.size())
      continue;
    const OperandQueueUnitBinding &unit = operandQueueUnits_[queue.unitBinding];
    CgraOperandQueueHeadDiagnostic diagnostic{
        queue.queue,
        queue.fu,
        unit.allocationUnit,
        unit.capacity,
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
    diagnostic.consumers.reserve(queue.consumers.size());
    for (const OperandQueueBinding::Consumer &consumer : queue.consumers)
      diagnostic.consumers.emplace_back(consumer.semanticActorOrdinal,
                                        consumer.inputOrdinal);
    if (!queue.entries.empty()) {
      const OperandQueueBinding::Entry &head = queue.entries.front();
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

} // namespace loom::sim::detail

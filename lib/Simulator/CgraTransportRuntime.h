#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H

#include "CgraComputeRuntime.h"
#include "CgraTransportStorageRuntime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::sim::detail {

struct CgraTokenPublication final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint64_t producerSequenceOrdinal = 0;
  Token token;
};

struct CgraTransportCompletion final {
  std::uint64_t semanticActorOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
};

struct CgraTransportFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraTokenPublication> publications;
  std::vector<CgraTransportCompletion> completions;
  std::vector<std::uint64_t> blockedTransfers;
};

/// Execution-local token transport for one mapped graph activation. It binds
/// exact Dataflow endpoints to dense DFG channel slots once; dynamic events
/// never search MLIR users or persistent-reference bytes.
class CgraTransportRuntime final {
public:
  static llvm::Expected<CgraTransportRuntime>
  create(const CgraFrozenExecutionPlan &plan,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
         SimulatorState &state, CgraPhysicalActionRuntime &physical);

  llvm::Error
  acceptActorEmissions(const SpatialEventCoordinate &coordinate,
                       llvm::MutableArrayRef<CgraActorEmission> emissions);

  llvm::Error acceptGraphIngressEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<GraphIngressEmission> emissions);

  llvm::Expected<std::vector<CgraTransportCompletion>>
  acceptPhysicalEvents(const CgraPhysicalLifecycleFrame &physicalFrame);

  llvm::Expected<CgraPhysicalTraceBinding>
  physicalTraceBinding(const CgraPhysicalLifecycleEvent &event) const;

  llvm::Expected<std::optional<CgraTransportFrame>> advance();

  llvm::Error retryBlocked(const SpatialEventCoordinate &coordinate);

  std::optional<SpatialEventCoordinate> nextCoordinate() const;

  bool hasPendingEvents() const {
    return !traversalEvents_.empty() || !storageEvents_.empty() ||
           !arrivalEvents_.empty() || !events_.empty() ||
           !requestedEvents_.empty() || activeTransferCount_ != 0;
  }
  bool hasBlockedTransfers() const { return blocked_.any(); }
  std::uint64_t activeTransferCount() const { return activeTransferCount_; }

private:
  enum class SinkKind : std::uint8_t { Channel, Observation };

  struct SinkBinding final {
    SinkKind kind = SinkKind::Channel;
    ChannelOrdinal channel = 0;
    mlir::Value observation;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
  };

  struct TransferBinding final {
    ::dataflow::CanonicalGraphProducerEndpointRef producer;
    std::uint64_t sinkOffset = 0;
    std::uint32_t sinkCount = 0;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
    std::uint64_t traversalNodeOffset = 0;
    std::uint32_t traversalNodeCount = 0;
    std::uint32_t traversalTerminalCount = 0;
    std::uint32_t consumedPhysicalUseCount = 0;
    std::optional<std::uint64_t> semanticActorOrdinal;
    std::uint64_t nextProducerSequenceOrdinal = 0;
    bool discard = false;
    bool active = false;
  };

  enum class TraversalNodeKind : std::uint8_t {
    PhysicalAction,
    BufferedStorage,
    RegisterStorageWrite,
    RegisterStorageRead,
  };

  struct TraversalNodeBinding final {
    TraversalNodeKind kind = TraversalNodeKind::PhysicalAction;
    std::uint64_t physicalUseOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t targetTraversalOffset = 0;
    std::uint32_t targetTraversalCount = 0;
    std::uint64_t successorOffset = 0;
    std::uint32_t successorCount = 0;
    std::uint32_t predecessorCount = 0;
    bool terminal = false;
  };

  struct InFlight final {
    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint64_t producerSequenceOrdinal = 0;
    Token token;
    bool arrivalScheduled = false;
    bool publicationScheduled = false;
    bool published = false;
    bool consumedRequested = false;
    std::uint32_t producedPermitted = 0;
    std::uint32_t producedRetired = 0;
    std::uint32_t traversalPermitted = 0;
    std::uint32_t traversalRetired = 0;
    std::uint32_t traversalTerminalsPermitted = 0;
    std::uint32_t consumedPermitted = 0;
    std::uint32_t consumedRetired = 0;
  };

  enum class ActionStage : std::uint8_t {
    Produced,
    Traversal,
    Storage,
    Consumed,
  };

  enum class StorageOperation : std::uint8_t {
    None,
    Enqueue,
    Dequeue,
    Simultaneous,
  };

  enum class ActionLifecycleState : std::uint8_t {
    Requested,
    Granted,
    Permitted,
    Retired,
  };

  struct ActionOwner final {
    std::uint64_t transferSlot = 0;
    std::uint64_t secondaryTransferSlot = invalidCgraTransportOrdinal;
    std::uint64_t traversalNodeOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t secondaryTraversalNodeOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
    ActionStage stage = ActionStage::Produced;
    StorageOperation storageOperation = StorageOperation::None;
    ActionLifecycleState state = ActionLifecycleState::Requested;
    std::uint64_t localActionOrdinal = 0;
  };

  struct PendingTransfer final {
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    Token *token = nullptr;
  };

  struct PendingActionTransfer final {
    std::uint64_t transferSlot = 0;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t traversalNodeOrdinal = invalidCgraTransportOrdinal;
  };

  enum class TraversalNodeState : std::uint8_t {
    Idle,
    Scheduled,
    Requested,
    WaitingStorage,
    Queued,
    Permitted,
  };

  struct StorageBinding final {
    StorageBinding(CgraTransportStorageRuntime state,
                   CgraTraversalStorageKind storageKind,
                   bool independentServices)
        : queue(std::move(state)), kind(storageKind),
          independentReadWriteServices(independentServices) {}

    CgraTransportStorageRuntime queue;
    CgraTraversalStorageKind kind = CgraTraversalStorageKind::None;
    std::uint64_t enqueueAction = invalidCgraTransportOrdinal;
    std::uint64_t dequeueAction = invalidCgraTransportOrdinal;
    std::uint64_t simultaneousAction = invalidCgraTransportOrdinal;
    std::vector<std::uint64_t> pendingEnqueueNodes;
    std::vector<std::uint64_t> pendingDequeueNodes;
    bool independentReadWriteServices = false;
    bool eventScheduled = false;
    std::uint8_t activeActionCount = 0;
  };

  struct StorageFrameCommit final {
    std::optional<CgraTransportStorageEntry> enqueue;
    std::optional<CgraTransportStorageEntry> expectedDequeue;
    std::uint64_t enqueueNode = invalidCgraTransportOrdinal;
    std::uint64_t dequeueNode = invalidCgraTransportOrdinal;
    std::uint8_t retireCount = 0;
    bool touched = false;
  };

  CgraTransportRuntime(
      const CgraFrozenExecutionPlan &plan, SimulatorState &state,
      CgraPhysicalActionRuntime &physical,
      std::vector<TransferBinding> bindings, std::vector<SinkBinding> sinks,
      std::vector<std::uint64_t> physicalUses,
      std::vector<TraversalNodeBinding> traversalNodes,
      std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets,
      std::vector<std::uint64_t> traversalSuccessors,
      std::vector<StorageBinding> storages,
      llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
          actorSourceBindings,
      llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings);

  std::uint64_t allocate(std::uint64_t bindingOrdinal,
                         std::uint64_t occurrenceOrdinal,
                         std::uint64_t producerSequenceOrdinal, Token token);
  llvm::Error acceptTransfers(const SpatialEventCoordinate &coordinate,
                              llvm::ArrayRef<PendingTransfer> transfers);
  llvm::Expected<std::vector<CgraPhysicalLifecycleEvent>>
  requestActions(llvm::ArrayRef<PendingActionTransfer> transfers,
                 ActionStage stage, const SpatialEventCoordinate &coordinate);
  llvm::Error scheduleArrival(std::uint64_t slot,
                              const SpatialEventCoordinate &coordinate);
  llvm::Expected<bool>
  scheduleReadyTraversals(std::uint64_t slot,
                          const SpatialEventCoordinate &coordinate);
  llvm::Error scheduleStorage(std::uint64_t storageOrdinal,
                              const SpatialEventCoordinate &coordinate);
  llvm::Error schedulePublication(std::uint64_t slot,
                                  const SpatialEventCoordinate &coordinate);
  std::optional<CgraTransportCompletion> maybeRelease(std::uint64_t slot);
  void scheduleAt(std::uint64_t slot,
                  const SpatialEventCoordinate &publicationCoordinate);
  bool canPublish(const TransferBinding &binding) const;
  void publish(std::uint64_t slot, CgraTransportFrame &frame);
  std::optional<CgraTransportCompletion> release(std::uint64_t slot);

  const CgraFrozenExecutionPlan *plan_ = nullptr;
  SimulatorState *state_ = nullptr;
  CgraPhysicalActionRuntime *physical_ = nullptr;
  std::vector<TransferBinding> bindings_;
  std::vector<SinkBinding> sinks_;
  std::vector<std::uint64_t> physicalUses_;
  std::vector<TraversalNodeBinding> traversalNodes_;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets_;
  std::vector<std::uint64_t> traversalSuccessors_;
  std::vector<StorageBinding> storages_;
  std::vector<StorageFrameCommit> storageFrameCommits_;
  std::vector<std::uint64_t> touchedStorageFrameCommits_;
  std::vector<std::uint32_t> traversalRemainingPredecessors_;
  std::vector<TraversalNodeState> traversalNodeStates_;
  std::vector<std::uint64_t> traversalNodeTransferSlots_;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings_;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings_;
  CgraEventQueue events_{"CGRA transport publication"};
  CgraEventQueue traversalEvents_{"CGRA traversal"};
  CgraEventQueue storageEvents_{"CGRA transport storage"};
  CgraEventQueue arrivalEvents_{"CGRA transport arrival"};
  CgraEventQueue requestedEvents_{"CGRA transport request"};
  std::vector<InFlight> inFlight_;
  std::vector<std::uint64_t> freeSlots_;
  llvm::SmallBitVector blocked_;
  std::vector<std::uint64_t> nextActionOccurrence_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, ActionOwner>
      actionOwners_;
  std::uint64_t activeTransferCount_ = 0;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H

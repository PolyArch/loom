#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H

#include "CgraComputeRuntime.h"
#include "CgraTransportGraph.h"
#include "CgraTransportStorageRuntime.h"
#include "Fabric/IR/TemporalOperandBuffer.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <deque>
#include <limits>
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
  llvm::SmallVector<CgraPhysicalLifecycleEvent, 8> physicalEvents;
  llvm::SmallVector<CgraTokenPublication, 4> publications;
  llvm::SmallVector<CgraTransportCompletion, 4> completions;
};

/// One resident token of a selected traversal storage. `queuePosition` is the
/// exact distance from the dequeue head, so position zero is the head that a
/// strict FIFO must retire before any later token can advance.
struct CgraStorageResidencyDiagnostic final {
  std::uint32_t queuePosition = 0;
  std::uint64_t bindingOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t occurrenceOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t traversalNodeOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
  /// The exact Physical Tag bit value the token carries. This is the semantic
  /// channel identity; `virtualChannelKey` is only its derived dense cache.
  llvm::APInt physicalTagValue = llvm::APInt(1, 0);
  /// Canonical channel identity of the token; see `tagVirtualChannelKey`.
  std::uint32_t virtualChannelKey = 0;
  std::uint64_t producerActorOrdinal = invalidCgraTransportOrdinal;
  /// Consumers this token still owes, as semantic actor input channels.
  std::vector<std::uint64_t> destinationChannelOrdinals;
  std::vector<std::uint64_t> destinationActorOrdinals;
  std::vector<std::uint32_t> destinationInputOrdinals;
};

struct CgraPendingTransferDiagnostic final {
  struct StorageHead final {
    std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t bindingOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t occurrenceOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t traversalNodeOrdinal = invalidCgraTransportOrdinal;
  };

  struct OperandQueueWait final {
    ::fabric::LogicalOperandQueueKey queue;
    ::loom::fabric::FabricFuOccurrenceRef fu;
    ::loom::fabric::FabricTransportEndpointRef ingress;
    llvm::APInt tag = llvm::APInt(1, 0);
    std::uint32_t allocationUnit = 0;
    std::uint32_t occupancy = 0;
    std::uint32_t reservations = 0;
    std::uint32_t capacity = 0;
  };

  std::uint64_t bindingOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint64_t producerActorOrdinal = invalidCgraTransportOrdinal;
  std::uint32_t producerResultOrdinal =
      std::numeric_limits<std::uint32_t>::max();
  /// The exact Physical Tag this token carries on its route, when tagged.
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
  llvm::APInt physicalTagValue = llvm::APInt(1, 0);
  std::optional<CgraPhysicalTagMappingOwner> physicalTagOwner;
  bool blocked = false;
  bool arrivalScheduled = false;
  bool publicationReady = false;
  bool published = false;
  bool consumedRequested = false;
  bool operandCapacityReserved = false;
  bool operandCapacityBlocked = false;
  std::uint32_t producedPermitted = 0;
  std::uint32_t producedRetired = 0;
  std::uint32_t traversalPermitted = 0;
  std::uint32_t traversalRetired = 0;
  std::uint32_t traversalTerminalsPermitted = 0;
  std::uint32_t consumedPermitted = 0;
  std::uint32_t consumedRetired = 0;
  std::uint32_t readySinkCount = 0;
  std::uint32_t publishedSinkCount = 0;
  std::uint32_t sinkCount = 0;
  std::uint32_t publicationCount = 0;
  std::uint32_t requestedPublicationCount = 0;
  std::uint32_t publishedPublicationCount = 0;
  std::vector<std::uint64_t> unpublishedActorOrdinals;
  std::vector<std::uint32_t> unpublishedInputOrdinals;
  std::vector<std::uint64_t> unpublishedReadyTokenCounts;
  std::uint64_t blockingTraversalNodeOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t blockingStorageOrdinal = invalidCgraTransportOrdinal;
  std::optional<::loom::fabric::FabricFifoOccurrenceRef> blockingFifoOccurrence;
  std::uint32_t blockingStorageOccupancy = 0;
  std::uint32_t blockingStorageReservations = 0;
  std::uint32_t blockingStorageCapacity = 0;
  std::optional<StorageHead> blockingStorageHead;
  bool blockingTraversalWaitingForStorage = false;
  std::uint32_t blockingDownstreamStorageCount = 0;
  std::uint32_t blockingUnbufferedSinkCount = 0;
  std::uint64_t blockingDownstreamStorageOrdinal = invalidCgraTransportOrdinal;
  std::uint32_t blockingDownstreamStorageOccupancy = 0;
  std::uint32_t blockingDownstreamStorageReservations = 0;
  std::uint32_t blockingDownstreamStorageCapacity = 0;
  bool blockingDownstreamStorageReserved = false;
  std::optional<StorageHead> blockingDownstreamStorageHead;
  std::uint64_t blockingActorOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t blockingReadyTokenCount = 0;
  std::uint64_t blockingQueueOccupancy = 0;
  std::uint64_t blockingQueueReservations = 0;
  std::uint64_t blockingQueueCapacity = 0;
  std::vector<OperandQueueWait> operandQueueWaits;
  std::optional<::dataflow::CanonicalGraphProducerEndpointRef> producer;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> blockingTraversals;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef>
      blockingDownstreamTraversals;
};

/// Terminal arbitration witness of one per-tag virtual-channel storage. Every
/// resident channel has been presented once since the last enqueue or dequeue
/// commit and every offer was refused, so the offer cursor completed one full
/// rotation without a grant. The queue schedules no further OfferAdvance until
/// an external event (a queue commit, released downstream capacity, or a
/// consumer publication) changes readiness; when no such event is pending the
/// rotation is arbitration progress that cannot become token progress, and
/// the execution is a closed wait rather than an infinite cursor rotation.
struct CgraStorageOfferRotationDiagnostic final {
  std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
  std::optional<::loom::fabric::FabricFifoOccurrenceRef> fifoOccurrence;
  std::uint32_t residentChannelCount = 0;
  std::uint32_t refusedOffersSinceCommit = 0;
  std::uint32_t occupancy = 0;
  std::uint32_t capacity = 0;
  /// Exact Physical Tag bit values of the resident channels in canonical
  /// unsigned ascending order, the order the rotation presented them.
  std::vector<llvm::APInt> residentTagValues;
};

/// Exact runtime witness for one selected Temporal PE operand queue. The
/// queue head is tracked as a producer binding/occurrence/sequence tuple, so
/// a closed wait can be joined to the Mapping-owned qualified pairing domain
/// without inferring progress from aggregate occupancy alone.
struct CgraOperandQueueHeadDiagnostic final {
  ::fabric::LogicalOperandQueueKey queue;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  std::uint32_t allocationUnit = 0;
  std::uint32_t capacity = 0;
  std::uint32_t occupancy = 0;
  std::uint32_t reservations = 0;
  std::uint64_t headBindingOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t headOccurrenceOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t headProducerSequenceOrdinal = invalidCgraTransportOrdinal;
  llvm::APInt headTag = llvm::APInt(1, 0);
  bool exactHead = false;
  std::vector<std::pair<std::uint64_t, unsigned>> consumers;
};

/// Token transport state for one mapped graph activation. The caller retains
/// the frozen plan and graph at stable addresses throughout the runtime's
/// lifetime; dynamic events use their dense bindings.
class CgraTransportRuntime final {
public:
  static llvm::Expected<CgraTransportRuntime>
  create(const CgraFrozenExecutionPlan &plan, const CgraTransportGraph &graph,
         SimulatorState &state, CgraPhysicalActionRuntime &physical);

  llvm::Error
  acceptActorEmissions(const SpatialEventCoordinate &coordinate,
                       llvm::MutableArrayRef<CgraActorEmission> emissions);

  llvm::Error acceptGraphIngressEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<GraphIngressEmission> emissions);

  llvm::Expected<bool> canAcceptGraphIngress(unsigned argumentOrdinal) const;

  /// Applies schema-owned input removals to declared operand storage and
  /// acknowledges unbuffered presentations at the actual consumer transition.
  /// Returned completions release only producers whose handoffs are complete.
  llvm::Expected<std::vector<CgraTransportCompletion>>
  acceptActorCommits(llvm::ArrayRef<CgraActorLifecycleEvent> events);

  /// Samples seeded durable queues before any launch event advances.
  llvm::Error initializeActivity(const SpatialEventCoordinate &coordinate);

  bool actorSourcesAvailable(std::uint64_t semanticActorOrdinal) const;

  llvm::Expected<std::vector<CgraTransportCompletion>>
  acceptPhysicalEvents(const CgraPhysicalLifecycleFrameView &physicalFrame);

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
  const ::loom::mapping::SpatialPeOperandProgressFeedback &
  operandQueueProgress() const;
  std::vector<CgraPendingTransferDiagnostic> pendingTransferDiagnostics() const;
  std::vector<CgraOperandQueueHeadDiagnostic>
  pendingOperandQueueHeadDiagnostics() const;
  /// Resident tokens of one selected traversal storage in dequeue order, with
  /// the transfer, tag, and consumer each token is continuing toward. This is
  /// the raw dynamic residency the closed-wait certificate projects; it adds
  /// no identity the runtime does not already own.
  std::vector<CgraStorageResidencyDiagnostic>
  storageResidencyDiagnostics(std::uint64_t storageOrdinal) const;
  /// Every non-empty virtual-channel storage whose refused-offer rotation is
  /// exhausted: each resident channel was presented and refused since the
  /// last commit, so the storage sleeps until an external event. At
  /// quiescence this is the typed no-complement terminal witness.
  std::vector<CgraStorageOfferRotationDiagnostic>
  exhaustedOfferRotationDiagnostics() const;

  /// Virtual-channel identity of one plan Physical Tag ordinal: the dense
  /// rank of its canonical tag value among the distinct values of the plan.
  /// Two ordinals that carry the same tag value share one channel, because
  /// the value is what the wire and a hardware arbiter observe. An ordinal
  /// outside the plan's tag inventory is invalid input, not channel zero.
  std::uint32_t tagVirtualChannelKey(std::uint64_t physicalTagOrdinal) const {
    assert(physicalTagOrdinal < graph_.tagVirtualChannelKeys.size() &&
           "Physical Tag ordinal outside the plan inventory");
    return graph_.tagVirtualChannelKeys[physicalTagOrdinal];
  }

  /// The complete rank cache, so a cold verifier can rebuild it from the
  /// plan's Physical Tag values with `internPhysicalTagChannelRanks` and
  /// compare rather than trust.
  llvm::ArrayRef<std::uint32_t> tagVirtualChannelRanks() const {
    return graph_.tagVirtualChannelKeys;
  }

  /// The next index of one channel's dense arrival sequence, or absent when
  /// the channel is outside the runtime's domain.
  std::optional<std::uint64_t>
  channelArrivalCount(std::uint64_t channelOrdinal) const {
    if (channelOrdinal >= channelArrivalCounts_.size())
      return std::nullopt;
    return channelArrivalCounts_[channelOrdinal];
  }

  /// The number of selected traversal storages.
  std::uint64_t storageCount() const { return storages_.size(); }

  /// The dequeue scheduling discipline of one traversal storage. Register
  /// storages have no discipline; they present their single global queue.
  std::optional<::fabric::FifoQueueDiscipline>
  storageQueueDiscipline(std::uint64_t storageOrdinal) const {
    if (storageOrdinal >= storages_.size() ||
        storages_[storageOrdinal].binding.kind !=
            CgraTraversalStorageKind::BufferedFifo)
      return std::nullopt;
    return storages_[storageOrdinal].queue.discipline();
  }

private:
  using SinkKind = CgraTransportGraph::SinkKind;
  using SinkBinding = CgraTransportGraph::SinkBinding;
  using PublicationBinding = CgraTransportGraph::PublicationBinding;
  using TransferBinding = CgraTransportGraph::TransferBinding;
  using TraversalNodeKind = CgraTransportGraph::TraversalNodeKind;
  using TraversalNodeBinding = CgraTransportGraph::TraversalNodeBinding;
  using OperandBufferBinding = CgraTransportGraph::OperandBufferBinding;

  enum class TraversalNodeState : std::uint8_t {
    Idle,
    Scheduled,
    Requested,
    WaitingStorage,
    Queued,
    Permitted,
  };

  struct TraversalState final {
    std::uint32_t remainingPredecessors = 0;
    TraversalNodeState state = TraversalNodeState::Idle;
    bool storageReserved = false;
  };

  struct TraversalOccurrence final {
    std::uint64_t transferSlot = 0;
    std::uint64_t nodeOrdinal = 0;

    bool operator==(const TraversalOccurrence &other) const {
      return transferSlot == other.transferSlot &&
             nodeOrdinal == other.nodeOrdinal;
    }
  };

  struct InFlight final {
    struct PublicationState final {
      bool consumedRequested = false;
      bool capacityReserved = false;
      bool capacityBlocked = false;
      bool enqueueCommitted = false;
      bool published = false;
      std::uint32_t consumedPermitted = 0;
      std::uint32_t consumedRetired = 0;
    };

    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint64_t producerSequenceOrdinal = 0;
    Token token;
    bool arrivalScheduled = false;
    bool publicationScheduled = false;
    bool publicationReady = false;
    bool published = false;
    bool consumedRequested = false;
    std::vector<bool> publishedSinks;
    std::uint32_t publishedSinkCount = 0;
    std::vector<bool> acceptedSinks;
    std::uint32_t acceptedSinkCount = 0;
    bool producerCompletionReported = false;
    std::vector<std::uint32_t> permittedSinkTerminals;
    std::vector<bool> readySinks;
    std::uint32_t readySinkCount = 0;
    std::vector<PublicationState> publications;
    /// Mutable route state belongs to this token occurrence. The selected DAG
    /// remains shared while earlier occurrences reside beyond durable storage.
    std::vector<TraversalState> traversals;
    /// Derived worklist: roots enter at allocation, successors only when their
    /// last predecessor is permitted. Scheduling consumes it in selected-node
    /// order.
    llvm::SmallVector<std::uint64_t, 4> readyTraversals;
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
    /// The internal arbitration transition of a virtual channel queue: one
    /// refused offer rotates the offer cursor at the commit boundary.
    OfferAdvance,
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
    std::uint64_t publicationBinding = invalidCgraTransportOrdinal;
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
    std::uint64_t publicationBinding = invalidCgraTransportOrdinal;
  };

  struct StorageState final {
    StorageState(const CgraTransportGraph::StorageBinding &binding,
                 CgraTransportStorageRuntime queue)
        : binding(binding), queue(std::move(queue)) {}

    const CgraTransportGraph::StorageBinding &binding;
    CgraTransportStorageRuntime queue;
    std::vector<TraversalOccurrence> pendingEnqueueNodes;
    std::vector<TraversalOccurrence> pendingDequeueNodes;
    bool eventScheduled = false;
    std::uint8_t activeActionCount = 0;
    std::uint32_t reservations = 0;
    /// A virtual-channel refusal rotates the cursor until every resident
    /// channel has been refused. Only a readiness change then wakes the queue.
    std::uint32_t offerRefusalsSinceCommit = 0;
  };

  struct StorageFrameCommit final {
    std::optional<CgraTransportStorageEntry> enqueue;
    std::optional<CgraTransportStorageEntry> expectedDequeue;
    std::uint64_t dequeueNode = invalidCgraTransportOrdinal;
    std::uint8_t retireCount = 0;
    bool touched = false;
  };

  struct ProducerState final {
    std::uint64_t nextProducerSequenceOrdinal = 0;
    bool sourceReserved = false;
    /// A result is pending until every sink accepts a durable handoff.
    bool producerPending = false;
  };

  struct OperandQueueUnitState final {
    explicit OperandQueueUnitState(
        const CgraTransportGraph::OperandQueueUnitBinding &binding)
        : binding(binding) {}

    const CgraTransportGraph::OperandQueueUnitBinding &binding;
    std::uint32_t occupancy = 0;
    std::uint32_t reservations = 0;
    std::optional<::loom::evaluation::ExactRatio> admissionCycle;
    std::uint32_t admissionCredits = 0;
  };

  struct OperandQueueState final {
    struct Entry final {
      std::uint64_t bindingOrdinal = invalidCgraTransportOrdinal;
      std::uint64_t occurrenceOrdinal = invalidCgraTransportOrdinal;
      std::uint64_t producerSequenceOrdinal = invalidCgraTransportOrdinal;
      llvm::APInt tag = llvm::APInt(1, 0);
    };

    explicit OperandQueueState(
        const CgraTransportGraph::OperandQueueBinding &binding)
        : binding(binding) {}

    const CgraTransportGraph::OperandQueueBinding &binding;
    std::uint32_t occupancy = 0;
    std::deque<Entry> entries;
  };

  CgraTransportRuntime(const CgraFrozenExecutionPlan &plan,
                       const CgraTransportGraph &graph, SimulatorState &state,
                       CgraPhysicalActionRuntime &physical);

  bool ownsTraversal(std::uint64_t slot, std::uint64_t nodeOrdinal) const;
  TraversalState &traversalState(std::uint64_t slot, std::uint64_t nodeOrdinal);
  const TraversalState &traversalState(std::uint64_t slot,
                                       std::uint64_t nodeOrdinal) const;
  void completeSource(InFlight &transfer);
  std::uint64_t allocate(std::uint64_t bindingOrdinal,
                         std::uint64_t occurrenceOrdinal,
                         std::uint64_t producerSequenceOrdinal, Token token);
  llvm::Error acceptTransfers(const SpatialEventCoordinate &coordinate,
                              llvm::ArrayRef<PendingTransfer> transfers);
  llvm::Expected<llvm::SmallVector<CgraPhysicalLifecycleEvent, 8>>
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
  llvm::Error beginOperandQueueCycle(const SpatialEventCoordinate &coordinate);
  struct OperandIngressAdmission final {
    ::fabric::OperandIngressAdmissionPriority priority =
        ::fabric::OperandIngressAdmissionPriority::Ordinary;
    llvm::SmallVector<::loom::mapping::SpatialPeOperandQualifiedPairingKey, 4>
        pairings;
  };
  llvm::Expected<OperandIngressAdmission>
  operandIngressAdmissionPriority(std::uint64_t slot,
                                  std::uint64_t publicationBinding) const;
  llvm::Expected<bool>
  reserveOperandQueueCapacity(std::uint64_t slot,
                              std::uint64_t publicationBinding,
                              const SpatialEventCoordinate &coordinate);
  llvm::Error
  observeOperandQueueActivity(std::uint64_t queueOrdinal,
                              const SpatialEventCoordinate &coordinate);
  llvm::Error
  commitOperandQueueEnqueue(std::uint64_t slot,
                            std::uint64_t publicationBinding,
                            const SpatialEventCoordinate &coordinate);
  std::optional<CgraTransportCompletion> maybeRelease(std::uint64_t slot);
  llvm::Expected<std::optional<CgraTransportCompletion>>
  maybeCompleteProducer(std::uint64_t slot);
  llvm::Error acceptDurableSinks(std::uint64_t slot,
                                 llvm::ArrayRef<std::uint32_t> localSinks);
  llvm::Expected<bool> markDirectSinksReady(std::uint64_t slot);
  llvm::Expected<bool> markTerminalSinksReady(std::uint64_t slot,
                                              std::uint64_t nodeOrdinal);
  void scheduleAt(std::uint64_t slot,
                  const SpatialEventCoordinate &publicationCoordinate);
  bool canPublish(std::uint64_t slot, std::uint64_t publicationBinding) const;
  bool canPublishSinks(const TransferBinding &binding,
                       bool operandCapacityReserved,
                       llvm::ArrayRef<std::uint32_t> localSinkOrdinals) const;
  bool canPublishSink(const SinkBinding &sink,
                      bool operandCapacityReserved) const;
  bool canAdvanceBufferedStorage(std::uint64_t slot,
                                 std::uint64_t nodeOrdinal) const;
  llvm::Error reserveDownstreamStorage(std::uint64_t slot,
                                       std::uint64_t nodeOrdinal);
  llvm::Error publish(std::uint64_t slot, std::uint64_t publicationBinding,
                      CgraTransportFrame &frame);
  std::optional<CgraTransportCompletion> release(std::uint64_t slot);

  const CgraFrozenExecutionPlan *plan_ = nullptr;
  SimulatorState *state_ = nullptr;
  CgraPhysicalActionRuntime *physical_ = nullptr;
  const CgraTransportGraph &graph_;
  std::vector<ProducerState> producerStates_;
  std::vector<StorageState> storages_;
  std::vector<OperandQueueUnitState> operandQueueUnits_;
  std::vector<OperandQueueState> operandQueues_;
  std::vector<StorageFrameCommit> storageFrameCommits_;
  std::vector<std::uint64_t> touchedStorageFrameCommits_;
  CgraEventQueue events_{"CGRA transport publication"};
  CgraEventQueue traversalEvents_{"CGRA traversal"};
  CgraEventQueue storageEvents_{"CGRA transport storage"};
  CgraEventQueue arrivalEvents_{"CGRA transport arrival"};
  CgraEventQueue requestedEvents_{"CGRA transport request"};
  std::vector<InFlight> inFlight_;
  std::vector<std::uint64_t> freeSlots_;
  /// Tokens delivered into each semantic channel by this transport. The value
  /// is the next index of the channel's dense arrival sequence, so a blocked
  /// input awaiting the channel's next token awaits exactly this producer
  /// occurrence.
  std::vector<std::uint64_t> channelArrivalCounts_;
  /// Indexed by in-flight transfer slot, including overlapping occurrences.
  llvm::SmallBitVector blocked_;
  std::vector<std::uint64_t> nextActionOccurrence_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, ActionOwner>
      actionOwners_;
  std::uint64_t activeTransferCount_ = 0;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H

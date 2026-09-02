#ifndef LOOM_SIMULATOR_CGRASIMULATOR_H
#define LOOM_SIMULATOR_CGRASIMULATOR_H

#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CgraExternalMemoryProvider.h"
#include "Simulator/CgraPhysicalTagOwner.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialExecutionSession.h"
#include "Simulator/SpatialTrace.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::sim {

enum class CgraUnsupportedMemoryContractKind : std::uint8_t {
  Volatile,
  AtomicAccess,
  AtomicRmw,
  CompareExchange,
  Fence,
};

struct CgraUnsupportedMemoryContract final {
  CgraUnsupportedMemoryContractKind kind;
  dataflow::ActorRef actor;
};

/// Typed refusal from the exact CGRA execution provider. Generic unsupported
/// errors cannot authorize a host fallback.
class CgraExecutionUnsupported final
    : public llvm::ErrorInfo<CgraExecutionUnsupported> {
public:
  static char ID;

  CgraExecutionUnsupported(CgraUnsupportedMemoryContract memoryContract,
                           std::string message)
      : memoryContract_(std::move(memoryContract)),
        message_(std::move(message)) {}

  const CgraUnsupportedMemoryContract &memoryContract() const {
    return memoryContract_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  CgraUnsupportedMemoryContract memoryContract_;
  std::string message_;
};

struct CgraSimulationCounters final {
  std::uint64_t eventFrameCount = 0;
  std::uint64_t actorCommitCount = 0;
  std::uint64_t actorRetirementCount = 0;
  std::uint64_t tokenPublicationCount = 0;
  std::uint64_t memoryLinearizationCount = 0;
  std::uint64_t physicalRequestCount = 0;
  std::uint64_t physicalGrantCount = 0;
  std::uint64_t physicalRetirementCount = 0;
  std::uint64_t emptyEventFrameCount = 0;
  std::uint64_t computeSourceFrameCount = 0;
  std::uint64_t memorySourceFrameCount = 0;
  std::uint64_t transportSourceFrameCount = 0;
  std::uint64_t physicalSourceFrameCount = 0;
  /// Runtime coordinate and arbitration observations. `delta` is retained
  /// only as an ordering diagnostic; it is never folded into cycle_count.
  std::uint64_t maximumReferenceCycleNumerator = 0;
  std::uint64_t maximumEventDelta = 0;
  std::uint64_t physicalGrantWaitCycleSum = 0;
  std::uint64_t physicalGrantWaitCycleMax = 0;
  std::uint64_t physicalActionLifetimeCycleSum = 0;
  std::uint64_t physicalActionLifetimeCycleMax = 0;
  std::uint64_t physicalGrantedLifetimeCycleSum = 0;
  std::uint64_t physicalGrantedLifetimeCycleMax = 0;
  std::uint64_t physicalGrantSameCycleCount = 0;
  std::uint64_t physicalGrantDelayedCount = 0;
  std::uint64_t nonIntegralTimingObservationCount = 0;
};

/// Attempt-local proof summary for a quiescent execution that cannot make
/// progress. Persistent Halted witnesses remain unavailable until the exact
/// FindingKind owner registers its terminal-witness schema.
struct CgraClosedWaitSetDiagnostic final {
  /// Exact immutable execution owners used to produce this witness. The
  /// witness is not a new artifact identity; these references let an
  /// independent Mapping/DSE verifier re-import the same owners.
  std::optional<CgraExecutionOwnerReferences> ownerReferences;

  struct OperandQueueHead final {
    ::fabric::LogicalOperandQueueKey queue;
    ::loom::fabric::FabricFuOccurrenceRef fu;
    std::uint32_t allocationUnit = 0;
    std::uint32_t capacity = 0;
    std::uint32_t occupancy = 0;
    std::uint32_t reservations = 0;
    std::uint64_t headBindingOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t headOccurrenceOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t headProducerSequenceOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    llvm::APInt headTag = llvm::APInt(1, 0);
    bool exactHead = false;
    std::vector<std::pair<std::uint64_t, unsigned>> consumers;
  };

  struct ActorFiring final {
    std::uint64_t semanticActorOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint32_t transitionCaseOrdinal = 0;
    std::uint32_t expectedTransfers = 0;
    std::uint32_t completedTransfers = 0;
    bool physicalComplete = false;
    bool causalReleaseSatisfied = false;
  };
  enum class ActorInputSourceKind : std::uint8_t {
    GraphInput,
    ActorResult,
    Unknown,
  };
  struct BlockedActorInput final {
    std::uint64_t semanticActorOrdinal = 0;
    std::uint64_t actorEntityId = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t inputOrdinal = 0;
    std::uint64_t channelOrdinal = std::numeric_limits<std::uint64_t>::max();
    ActorInputSourceKind sourceKind = ActorInputSourceKind::Unknown;
    std::uint64_t definingActorOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t definingActorEntityId =
        std::numeric_limits<std::uint64_t>::max();
    bool definingActorTerminal = false;
    /// The exact producer occurrence the blocked input awaits: the next index
    /// in the channel's dense arrival sequence. Equal to the producer result
    /// occurrence because one firing appends exactly one token to the channel.
    std::uint64_t expectedProducerOccurrenceOrdinal =
        std::numeric_limits<std::uint64_t>::max();
  };
  struct PhysicalAction final {
    std::uint64_t actionOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint8_t clientKind = 0;
    std::optional<std::uint64_t> semanticActorOrdinal;
    bool granted = false;
    bool hasCommit = false;
    bool requiresCausalRelease = false;
    bool intrinsicReleaseReached = false;
    bool causalReleaseReached = false;
  };
  struct Transfer final {
    struct StorageHead final {
      std::uint64_t storageOrdinal = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t bindingOrdinal = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t occurrenceOrdinal =
          std::numeric_limits<std::uint64_t>::max();
      std::uint64_t traversalNodeOrdinal =
          std::numeric_limits<std::uint64_t>::max();
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
    std::uint64_t producerActorOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint32_t producerResultOrdinal =
        std::numeric_limits<std::uint32_t>::max();
    /// The exact Physical Tag this token carries on its route, when tagged.
    std::uint64_t physicalTagOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    llvm::APInt physicalTagValue = llvm::APInt(1, 0);
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
    std::uint64_t blockingTraversalNodeOrdinal = 0;
    std::uint64_t blockingStorageOrdinal = 0;
    std::optional<::loom::fabric::FabricFifoOccurrenceRef>
        blockingFifoOccurrence;
    std::uint32_t blockingStorageOccupancy = 0;
    std::uint32_t blockingStorageReservations = 0;
    std::uint32_t blockingStorageCapacity = 0;
    std::optional<StorageHead> blockingStorageHead;
    bool blockingTraversalWaitingForStorage = false;
    std::uint32_t blockingDownstreamStorageCount = 0;
    std::uint32_t blockingUnbufferedSinkCount = 0;
    std::uint64_t blockingDownstreamStorageOrdinal = 0;
    std::uint32_t blockingDownstreamStorageOccupancy = 0;
    std::uint32_t blockingDownstreamStorageReservations = 0;
    std::uint32_t blockingDownstreamStorageCapacity = 0;
    bool blockingDownstreamStorageReserved = false;
    std::optional<StorageHead> blockingDownstreamStorageHead;
    std::uint64_t blockingActorOrdinal = 0;
    std::uint64_t blockingReadyTokenCount = 0;
    std::uint64_t blockingQueueOccupancy = 0;
    std::uint64_t blockingQueueReservations = 0;
    std::uint64_t blockingQueueCapacity = 0;
    std::vector<OperandQueueWait> operandQueueWaits;
    std::optional<::dataflow::CanonicalGraphProducerEndpointRef> producer;
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> blockingTraversals;
    std::vector<::loom::fabric::FabricPhysicalTraversalRef>
        blockingDownstreamTraversals;
    std::optional<CgraPhysicalTagMappingOwner> physicalTagOwner;
  };
  enum class TransferWaitKind : std::uint8_t {
    ActorPublication,
    StorageHead,
    DownstreamStorageHead,
  };
  struct TransferWaitCycleEdge final {
    std::uint64_t waitingBindingOrdinal = 0;
    std::uint64_t waitingOccurrenceOrdinal = 0;
    std::uint64_t blockingActorOrdinal = 0;
    std::uint64_t blockingBindingOrdinal = 0;
    std::uint64_t blockingOccurrenceOrdinal = 0;
    TransferWaitKind kind = TransferWaitKind::ActorPublication;
  };
  enum class ActorWaitKind : std::uint8_t {
    OutputBackpressure,
    MissingInput,
  };
  struct ActorWaitCycleEdge final {
    std::uint64_t waitingActorOrdinal = 0;
    std::uint64_t blockingActorOrdinal = 0;
    ActorWaitKind kind = ActorWaitKind::MissingInput;
  };
  std::uint64_t pendingActorFirings = 0;
  std::uint64_t pendingTransfers = 0;
  std::uint64_t pendingPhysicalActions = 0;
  bool graphRetirementVisible = false;
  std::vector<ActorFiring> actorFirings;
  /// Exact missing inputs selected by the current semantic state of every
  /// blocked actor. Producer terminality distinguishes finite underproduction
  /// from an unresolved internal wait without creating another liveness owner.
  std::vector<BlockedActorInput> blockedActorInputs;
  std::vector<Transfer> transfers;
  std::vector<PhysicalAction> physicalActions;
  /// Canonical cycle in the actual quiescent transfer wait-for graph. An edge
  /// says that one blocked transfer needs a consumer actor whose active
  /// firing cannot retire before another pending transfer completes. Empty
  /// means this narrower finite-buffer cycle was not established; it does not
  /// turn a quiescent execution into a successful one.
  std::vector<TransferWaitCycleEdge> transferWaitCycle;
  /// Canonical cycle in the actor-level wait closure. Output-backpressure
  /// edges are reconstructed from blocked physical transfers. Missing-input
  /// edges use only the dynamic transition selected by the actor semantics and
  /// are admitted when its missing producers remain in the greatest closed set.
  std::vector<ActorWaitCycleEdge> actorWaitCycle;

  /// Unified causal certificate of one quiescent closed wait. Nodes are typed
  /// dynamic owners — one exact actor firing occurrence, or one queue class of
  /// one physical storage — and every edge quotes one dynamic wait fact the
  /// runtime observed. The certificate is the single closed strongly connected
  /// component of that combined wait-for relation, so it stays bounded by the
  /// closure that actually deadlocked. An independent Mapping or DSE owner can
  /// rebuild the closed cycle from these edges alone; the runtime remains the
  /// only owner of the dynamic facts they quote.
  struct WaitActorFiringKey final {
    std::uint64_t semanticActorOrdinal = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t occurrenceOrdinal = std::numeric_limits<std::uint64_t>::max();

    friend bool operator==(const WaitActorFiringKey &lhs,
                           const WaitActorFiringKey &rhs) {
      return lhs.semanticActorOrdinal == rhs.semanticActorOrdinal &&
             lhs.occurrenceOrdinal == rhs.occurrenceOrdinal;
    }
    friend bool operator<(const WaitActorFiringKey &lhs,
                          const WaitActorFiringKey &rhs) {
      return std::tie(lhs.semanticActorOrdinal, lhs.occurrenceOrdinal) <
             std::tie(rhs.semanticActorOrdinal, rhs.occurrenceOrdinal);
    }
  };
  /// The queue class of one storage a wait refers to. A StrictFifo queue has
  /// one global class; a PerTagVirtualChannel queue has one class per resident
  /// Physical Tag value, identified by the exact tag bits rather than any
  /// plan-local ordinal.
  struct WaitQueueClass final {
    bool tagLocal = false;
    llvm::APInt tagValue = llvm::APInt(1, 0);

    static WaitQueueClass global() { return {}; }
    static WaitQueueClass tag(llvm::APInt value) {
      return WaitQueueClass{true, std::move(value)};
    }

    friend bool operator==(const WaitQueueClass &lhs,
                           const WaitQueueClass &rhs) {
      return lhs.tagLocal == rhs.tagLocal &&
             ::fabric::comparePhysicalTagValues(lhs.tagValue, rhs.tagValue) ==
                 0;
    }
    friend bool operator!=(const WaitQueueClass &lhs,
                           const WaitQueueClass &rhs) {
      return !(lhs == rhs);
    }
    friend bool operator<(const WaitQueueClass &lhs,
                          const WaitQueueClass &rhs) {
      if (lhs.tagLocal != rhs.tagLocal)
        return !lhs.tagLocal;
      return ::fabric::comparePhysicalTagValues(lhs.tagValue, rhs.tagValue) <
             0;
    }
  };
  enum class WaitStorageDomain : std::uint8_t {
    TraversalStorage,
    OperandQueue,
  };
  struct WaitStorageQueueKey final {
    WaitStorageDomain domain = WaitStorageDomain::TraversalStorage;
    std::uint64_t ordinal = std::numeric_limits<std::uint64_t>::max();
    WaitQueueClass queueClass;

    friend bool operator==(const WaitStorageQueueKey &lhs,
                           const WaitStorageQueueKey &rhs) {
      return lhs.domain == rhs.domain && lhs.ordinal == rhs.ordinal &&
             lhs.queueClass == rhs.queueClass;
    }
    friend bool operator<(const WaitStorageQueueKey &lhs,
                          const WaitStorageQueueKey &rhs) {
      return std::tie(lhs.domain, lhs.ordinal, lhs.queueClass) <
             std::tie(rhs.domain, rhs.ordinal, rhs.queueClass);
    }
  };
  struct WaitOwnerKey final {
    std::variant<WaitActorFiringKey, WaitStorageQueueKey> owner;

    friend bool operator==(const WaitOwnerKey &lhs, const WaitOwnerKey &rhs) {
      return lhs.owner == rhs.owner;
    }
    friend bool operator<(const WaitOwnerKey &lhs, const WaitOwnerKey &rhs) {
      if (lhs.owner.index() != rhs.owner.index())
        return lhs.owner.index() < rhs.owner.index();
      if (lhs.owner.index() == 0)
        return std::get<0>(lhs.owner) < std::get<0>(rhs.owner);
      return std::get<1>(lhs.owner) < std::get<1>(rhs.owner);
    }
  };
  enum class WaitEdgeKind : std::uint8_t {
    /// An actor firing cannot complete an input until a producer firing
    /// supplies it; the edge carries the expected producer occurrence.
    ActorMissingInput,
    /// An actor firing cannot durably accept its output transfer because a
    /// storage queue cannot admit it.
    ActorOutputBackpressure,
    /// An actor firing's awaited token is resident behind the head of its
    /// queue class — the strict FIFO head, or its tag-local head.
    StorageOrder,
    /// A storage queue's class head cannot continue into the downstream
    /// storage queue, which is full at cycle start.
    StorageDownstream,
    /// A storage queue's class head reached the route terminal but the
    /// consumer actor input has not taken it.
    StorageConsumer,
    /// An actor firing's input wait joins at the exact operand queue head.
    OperandQueueWait,
  };
  struct WaitEdge final {
    WaitOwnerKey from;
    WaitOwnerKey to;
    WaitEdgeKind kind = WaitEdgeKind::ActorMissingInput;
    /// Waiting side, when the waiting node is an actor firing.
    std::uint32_t waitingInputOrdinal =
        std::numeric_limits<std::uint32_t>::max();
    std::uint64_t waitingChannelOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    /// The in-flight transfer this edge is about, when one exists.
    std::uint64_t bindingOrdinal = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t occurrenceOrdinal = std::numeric_limits<std::uint64_t>::max();
    /// Storage facts, when the edge crosses a physical storage.
    std::uint64_t storageOrdinal = std::numeric_limits<std::uint64_t>::max();
    std::optional<::loom::fabric::FabricFifoOccurrenceRef> fifoOccurrence;
    std::uint32_t storageCapacity = 0;
    std::uint32_t storageOccupancy = 0;
    /// Queue position of the awaited token inside its queue class and the
    /// class head it waits behind, with the exact tag bit values on both.
    std::uint32_t awaitedClassPosition =
        std::numeric_limits<std::uint32_t>::max();
    std::optional<llvm::APInt> awaitedTagValue;
    std::optional<llvm::APInt> headTagValue;
    std::uint64_t headBindingOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t headOccurrenceOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t headDestinationActorOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint32_t headDestinationInputOrdinal =
        std::numeric_limits<std::uint32_t>::max();
    std::uint64_t headDestinationChannelOrdinal =
        std::numeric_limits<std::uint64_t>::max();
  };
  /// A closed wait that carries no exact certificate states why. The runtime
  /// never forges a certificate it cannot establish from dynamic owners.
  enum class WaitProofFailure : std::uint8_t {
    /// A required occurrence or dynamic owner could not be determined.
    IndeterminateDynamicOwner,
    /// The combined wait-for relation holds no closed strongly connected
    /// component whose every node waits inside the component.
    NoClosedComponent,
  };
  std::optional<WaitProofFailure> waitProofFailure;
  /// Empty when no closed strongly connected wait component was established.
  /// A non-empty certificate is a proof obligation for the Mapping owner, not
  /// a diagnostic convenience. Every node of the certificate has at least one
  /// edge inside the component, and the component is a sink of the wait-for
  /// relation. JSON and feedback projections derive from this typed form; it
  /// never becomes a second semantic owner.
  std::vector<WaitEdge> waitCertificate;
  /// Shared derived Mapping/Simulator queue projection summary. These fields
  /// carry no new runtime identity and are absent when no operand queues are
  /// selected.
  std::uint64_t operandQueueGroupCount = 0;
  std::uint64_t operandQueuePotentiallyBlockingGroupCount = 0;
  std::uint64_t operandQueueSharedIngressPressure = 0;
  std::uint64_t operandQueueDistinctIngressCount = 0;
  std::uint64_t operandQueuePairingKeyCount = 0;
  std::uint8_t operandQueueProgressStatus = 0;
  std::uint8_t operandQueueProgressSupport = 0;
  std::optional<::loom::ComponentViewDigest> operandQueueProjectionDigest;
  std::vector<OperandQueueHead> operandQueueHeads;
  /// Terminal witness of one per-tag virtual-channel storage: every resident
  /// channel was presented once since the last queue commit and refused, so
  /// the offer cursor completed a full rotation without a grant and the queue
  /// sleeps until an external event. At quiescence no such event remains, so
  /// the refused class heads in the certificate are exact, not the artifact
  /// of an offer the port never made.
  struct OfferRotationWitness final {
    std::uint64_t storageOrdinal = std::numeric_limits<std::uint64_t>::max();
    std::optional<::loom::fabric::FabricFifoOccurrenceRef> fifoOccurrence;
    std::uint32_t residentChannelCount = 0;
    std::uint32_t refusedOffersSinceCommit = 0;
    std::uint32_t occupancy = 0;
    std::uint32_t capacity = 0;
    std::vector<llvm::APInt> residentTagValues;
  };
  std::vector<OfferRotationWitness> exhaustedOfferRotations;
};

struct RetiredCgraSimulation final {
  SpatialFunctionalObservations observations;
  SpatialProgressObservations progress;
  CgraSimulationCounters counters;
};

struct HaltedCgraSimulation final {
  SpatialFunctionalObservations observations;
  SpatialProgressObservations progress;
  CgraSimulationCounters counters;
};

struct CgraSimulationOutcome final {
  SpatialExecutionSessionState state = SpatialExecutionSessionState::Failed;
  CgraSimulationCounters counters;
  std::optional<RetiredCgraSimulation> retired;
  std::optional<HaltedCgraSimulation> halted;
  std::optional<CgraClosedWaitSetDiagnostic> closedWaitSet;
};

/// Independent closure check of one emitted closed-wait certificate: every
/// typed owner it names waits inside the certificate and the whole
/// certificate is one strongly connected component. Returns false when the
/// certificate is absent or a proof failure was reported.
bool verifyClosedWaitCertificateClosure(
    const CgraClosedWaitSetDiagnostic &closedWait);

/// The shared closure predicate over the certificate's semantic edge slice.
/// Diagnostic and durable certificate owners both delegate to this function.
bool verifyClosedWaitCertificateClosure(
    llvm::ArrayRef<CgraClosedWaitSetDiagnostic::WaitEdge> edges);

/// Removable invocation-local admission projection for repeated execution of
/// one exact workload/runtime-input pair. The projection shares ownership of
/// its immutable PreparedCgraExecution closure.
class PreparedCgraWorkloadExecution final {
public:
  PreparedCgraWorkloadExecution(PreparedCgraWorkloadExecution &&) noexcept;
  PreparedCgraWorkloadExecution &
  operator=(PreparedCgraWorkloadExecution &&) noexcept;
  ~PreparedCgraWorkloadExecution();

  PreparedCgraWorkloadExecution(const PreparedCgraWorkloadExecution &) = delete;
  PreparedCgraWorkloadExecution &
  operator=(const PreparedCgraWorkloadExecution &) = delete;

private:
  struct Impl;
  explicit PreparedCgraWorkloadExecution(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend llvm::Expected<PreparedCgraWorkloadExecution>
  prepareCgraWorkloadExecution(const PreparedCgraExecution &,
                               const CanonicalSimulationWorkload &,
                               const CanonicalSimulationRuntimeInput &);
  friend llvm::Expected<CgraExecutionSession>
  startCgraExecutionSession(const PreparedCgraWorkloadExecution &,
                            const CanonicalSimulationWorkload &,
                            const CanonicalSimulationRuntimeInput &,
                            std::optional<TraceCaptureLevel>,
                            CgraExternalMemoryProvider *);
};

llvm::Expected<PreparedCgraWorkloadExecution> prepareCgraWorkloadExecution(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

class CgraExecutionSession final {
public:
  CgraExecutionSession(CgraExecutionSession &&) noexcept;
  CgraExecutionSession &operator=(CgraExecutionSession &&) noexcept;
  ~CgraExecutionSession();

  CgraExecutionSession(const CgraExecutionSession &) = delete;
  CgraExecutionSession &operator=(const CgraExecutionSession &) = delete;

  SpatialExecutionSessionState state() const;
  const CgraSimulationCounters &counters() const;
  const std::optional<CgraClosedWaitSetDiagnostic> &closedWaitSet() const;
  const std::optional<SpatialDiagnosticTrace> &diagnosticTrace() const;

  llvm::Expected<SpatialExecutionSessionState> advance(
      std::uint64_t maxEventFrames,
      std::optional<std::chrono::steady_clock::time_point> executionDeadline =
          std::nullopt);

  llvm::Expected<RetiredCgraSimulation> takeRetiredSimulation();
  llvm::Expected<HaltedCgraSimulation> takeHaltedSimulation();

private:
  struct Impl;
  explicit CgraExecutionSession(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend llvm::Expected<CgraExecutionSession> startCgraExecutionSession(
      const PreparedCgraExecution &, const CanonicalSimulationWorkload &,
      const CanonicalSimulationRuntimeInput &, std::optional<TraceCaptureLevel>,
      CgraExternalMemoryProvider *);
  friend llvm::Expected<CgraExecutionSession>
  startCgraExecutionSession(const PreparedCgraWorkloadExecution &,
                            const CanonicalSimulationWorkload &,
                            const CanonicalSimulationRuntimeInput &,
                            std::optional<TraceCaptureLevel>,
                            CgraExternalMemoryProvider *);
  friend llvm::Expected<CgraSimulationOutcome>
  simulateCgraWorkload(const PreparedCgraExecution &,
                       const CanonicalSimulationWorkload &,
                       const CanonicalSimulationRuntimeInput &, std::uint64_t,
                       std::optional<std::chrono::steady_clock::time_point>,
                       CgraExternalMemoryProvider *);
  friend llvm::Expected<CgraSimulationOutcome>
  simulateCgraWorkload(const PreparedCgraWorkloadExecution &,
                       const CanonicalSimulationWorkload &,
                       const CanonicalSimulationRuntimeInput &, std::uint64_t,
                       std::optional<std::chrono::steady_clock::time_point>,
                       CgraExternalMemoryProvider *);
};

llvm::Expected<CgraExecutionSession> startCgraExecutionSession(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::optional<TraceCaptureLevel> traceLevel = std::nullopt,
    CgraExternalMemoryProvider *externalMemoryProvider = nullptr);

llvm::Expected<CgraExecutionSession> startCgraExecutionSession(
    const PreparedCgraWorkloadExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::optional<TraceCaptureLevel> traceLevel = std::nullopt,
    CgraExternalMemoryProvider *externalMemoryProvider = nullptr);

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt,
    CgraExternalMemoryProvider *externalMemoryProvider = nullptr);

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraWorkloadExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt,
    CgraExternalMemoryProvider *externalMemoryProvider = nullptr);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRASIMULATOR_H

#ifndef LOOM_SIMULATOR_CGRASIMULATOR_H
#define LOOM_SIMULATOR_CGRASIMULATOR_H

#include "Simulator/CGRAAdmission.h"
#include "Simulator/CgraExternalMemoryProvider.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialExecutionSession.h"
#include "Simulator/SpatialTrace.h"
#include "Common/ComponentViewDigest.h"
#include "Fabric/IR/TemporalOperandBuffer.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::sim {

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
    std::uint64_t channelOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    ActorInputSourceKind sourceKind = ActorInputSourceKind::Unknown;
    std::uint64_t definingActorOrdinal =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t definingActorEntityId =
        std::numeric_limits<std::uint64_t>::max();
    bool definingActorTerminal = false;
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
      std::uint64_t storageOrdinal =
          std::numeric_limits<std::uint64_t>::max();
      std::uint64_t bindingOrdinal =
          std::numeric_limits<std::uint64_t>::max();
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
};

struct RetiredCgraSimulation final {
  SpatialFunctionalObservations observations;
  SpatialProgressObservations progress;
  CgraSimulationCounters counters;
};

struct CgraSimulationOutcome final {
  SpatialExecutionSessionState state = SpatialExecutionSessionState::Failed;
  CgraSimulationCounters counters;
  std::optional<RetiredCgraSimulation> retired;
  std::optional<CgraClosedWaitSetDiagnostic> closedWaitSet;
};

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

private:
  struct Impl;
  explicit CgraExecutionSession(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend llvm::Expected<CgraExecutionSession> startCgraExecutionSession(
      const PreparedCgraExecution &, const CanonicalSimulationWorkload &,
      const CanonicalSimulationRuntimeInput &, std::optional<TraceCaptureLevel>,
      CgraExternalMemoryProvider *);
  friend llvm::Expected<CgraSimulationOutcome>
  simulateCgraWorkload(const PreparedCgraExecution &,
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

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt,
    CgraExternalMemoryProvider *externalMemoryProvider = nullptr);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRASIMULATOR_H

#ifndef LOOM_SIMULATOR_CGRASIMULATOR_H
#define LOOM_SIMULATOR_CGRASIMULATOR_H

#include "Simulator/CGRAAdmission.h"
#include "Simulator/CgraExternalMemoryProvider.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialExecutionSession.h"
#include "Simulator/SpatialTrace.h"

#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
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
};

/// Attempt-local proof summary for a quiescent execution that cannot make
/// progress. Persistent Halted witnesses remain unavailable until the exact
/// FindingKind owner registers its terminal-witness schema.
struct CgraClosedWaitSetDiagnostic final {
  struct ActorFiring final {
    std::uint64_t semanticActorOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint32_t transitionCaseOrdinal = 0;
    std::uint32_t expectedTransfers = 0;
    std::uint32_t completedTransfers = 0;
    bool physicalComplete = false;
    bool causalReleaseSatisfied = false;
  };
  struct PhysicalAction final {
    std::uint64_t actionOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint8_t clientKind = 0;
    bool granted = false;
    bool hasCommit = false;
    bool requiresCausalRelease = false;
    bool intrinsicReleaseReached = false;
    bool causalReleaseReached = false;
  };
  struct Transfer final {
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
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
    std::uint32_t blockingStorageOccupancy = 0;
    std::uint32_t blockingStorageReservations = 0;
    std::uint32_t blockingStorageCapacity = 0;
    std::uint8_t blockingTraversalState = 0;
    std::uint32_t blockingDownstreamStorageCount = 0;
    std::uint32_t blockingUnbufferedSinkCount = 0;
    std::uint64_t blockingDownstreamStorageOrdinal = 0;
    std::uint32_t blockingDownstreamStorageOccupancy = 0;
    std::uint32_t blockingDownstreamStorageReservations = 0;
    std::uint32_t blockingDownstreamStorageCapacity = 0;
    bool blockingDownstreamStorageReserved = false;
    std::uint64_t blockingActorOrdinal = 0;
    std::uint64_t blockingReadyTokenCount = 0;
    std::uint64_t blockingQueueOccupancy = 0;
    std::uint64_t blockingQueueReservations = 0;
    std::uint64_t blockingQueueCapacity = 0;
  };
  std::uint64_t pendingActorFirings = 0;
  std::uint64_t pendingTransfers = 0;
  std::uint64_t pendingPhysicalActions = 0;
  bool graphRetirementVisible = false;
  std::vector<ActorFiring> actorFirings;
  std::vector<Transfer> transfers;
  std::vector<PhysicalAction> physicalActions;
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

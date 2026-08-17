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
};

/// Attempt-local proof summary for a quiescent execution that cannot make
/// progress. Persistent Halted witnesses remain unavailable until the exact
/// FindingKind owner registers its terminal-witness schema.
struct CgraClosedWaitSetDiagnostic final {
  std::uint64_t pendingActorFirings = 0;
  std::uint64_t pendingTransfers = 0;
  std::uint64_t pendingPhysicalActions = 0;
  bool graphRetirementVisible = false;
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

  friend llvm::Expected<CgraExecutionSession>
  startCgraExecutionSession(const PreparedCgraExecution &,
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

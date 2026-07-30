#ifndef LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H
#define LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>

namespace loom::sim {

struct NativeStructuredProgramObservations;

enum class SourceBackedDfgValidationStatus : std::uint8_t {
  Equivalent,
  Mismatch,
  Inapplicable,
};

/// Transient execution accounting for one exact source-backed replay. The
/// normalized semantic result belongs to EvaluationEvidence; these counters
/// are removable provider state and never become a candidate identity field.
struct SourceBackedDfgValidationResult final {
  SourceBackedDfgValidationStatus status =
      SourceBackedDfgValidationStatus::Inapplicable;
  std::uint64_t dynamicActivations = 0;
  std::uint64_t valueLanesCompared = 0;
  std::uint64_t memoryBytesCompared = 0;
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;
  double simulationSeconds = 0.0;
  std::map<dataflow::OperationSchemaId, std::uint64_t> operationFireCounts;
  std::optional<CanonicalValueSequence> sourceReturnValue;
};

struct SourceBackedDfgValidationLimits final {
  std::uint64_t maxWavefrontSteps;
  std::uint64_t maxEventCount;
  std::uint64_t maxRetainedCaptureBytes;
  std::chrono::steady_clock::duration maxSimulationWallTime =
      std::chrono::steady_clock::duration::max();
};

/// Reapply one exact parent-local Structured ownership decision, capture every
/// selected-region activation under the common source workload, replay the
/// mechanically derived Canonical Dataflow graph, and compare value, memory,
/// and retirement semantics. The source workload/runtime pair remains the
/// sole persistent input identity; activation-specific Spatial inputs are
/// derived transiently. A production semantic gate may provide its already
/// computed source observation so the selected execution also proves the
/// whole-program transformation without executing the candidate twice.
llvm::Expected<SourceBackedDfgValidationResult> validateSourceBackedDfgReplay(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const frontend::SpatialOwnershipScope &scope,
    const frontend::SpatialOwnershipDecisionPoint &decision,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    SourceBackedDfgValidationLimits limits,
    const NativeStructuredProgramObservations *sourceObservations = nullptr);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

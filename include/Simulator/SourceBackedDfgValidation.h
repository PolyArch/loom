#ifndef LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H
#define LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::sim {

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
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;
};

struct SourceBackedDfgValidationLimits final {
  std::uint64_t maxWavefrontSteps;
  std::uint64_t maxEventCount;
  std::uint64_t maxRetainedCaptureBytes;
};

/// Reapply one exact parent-local Structured ownership decision, capture every
/// selected-region activation under the common source workload, replay the
/// mechanically derived Canonical Dataflow graph, and compare value, memory,
/// and retirement semantics. The source workload/runtime pair remains the
/// sole persistent input identity; activation-specific Spatial inputs are
/// derived transiently.
llvm::Expected<SourceBackedDfgValidationResult> validateSourceBackedDfgReplay(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const frontend::SpatialOwnershipScope &scope,
    const frontend::SpatialOwnershipDecisionPoint &decision,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    SourceBackedDfgValidationLimits limits);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

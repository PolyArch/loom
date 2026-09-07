#ifndef LOOM_DSE_STRUCTUREDFUNCTIONALREPLAYBUDGET_H
#define LOOM_DSE_STRUCTUREDFUNCTIONALREPLAYBUDGET_H

#include "Common/ExecutionControl.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include <optional>

namespace loom::frontend {
struct MaterializedOwnershipCandidate;
class StructuredProgramCandidate;
} // namespace loom::frontend
namespace loom::sim {
class CanonicalSimulationWorkload;
class CanonicalSimulationRuntimeInput;
} // namespace loom::sim
namespace loom::dse {
class StructuredOwnershipSharedEvaluation;

/// Explicit work limits are aggregate caps. An absent dimension is planned
/// once from the exact workload's observed selected-region activation count.
struct StructuredFunctionalReplayBudget final {
  std::optional<std::uint64_t> maxWavefrontSteps;
  std::optional<std::uint64_t> maxEventCount;
  std::uint64_t maxRetainedCaptureBytes = 256ULL * 1024ULL * 1024ULL;
  std::chrono::steady_clock::duration maxSimulationWallTime =
      std::chrono::steady_clock::duration::max();
};

llvm::Expected<sim::SourceBackedDfgValidationLimits>
planStructuredFunctionalReplayBudget(
    const StructuredFunctionalReplayBudget &budget,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const frontend::StructuredProgramCandidate &source,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation,
    ExecutionControlView executionControl);

} // namespace loom::dse
#endif

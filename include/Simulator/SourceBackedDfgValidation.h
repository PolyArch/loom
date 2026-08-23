#ifndef LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H
#define LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredExecutionShape.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <vector>

namespace loom::sim {

struct NativeStructuredProgramObservations;

enum class SourceBackedDfgValidationStatus : std::uint8_t {
  Equivalent,
  Mismatch,
  Inapplicable,
};

/// Persistent identities of one exact activation input captured by the
/// source-backed replay owner. The workload and runtime-input Artifacts remain
/// the semantic owners; this pair is invocation provenance used by downstream
/// execution validation.
struct SourceBackedDfgReplayCaseReference final {
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;

  friend bool operator==(const SourceBackedDfgReplayCaseReference &lhs,
                         const SourceBackedDfgReplayCaseReference &rhs) {
    return lhs.workload == rhs.workload &&
           lhs.runtimeInput == rhs.runtimeInput;
  }
};

using SourceBackedDfgReplayCasePublisher = std::function<
    llvm::Expected<SourceBackedDfgReplayCaseReference>(
        const CanonicalSimulationWorkload &,
        const CanonicalSimulationRuntimeInput &)>;

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
  std::vector<SourceBackedDfgReplayCaseReference> replayCases;
};

struct SourceBackedDfgValidationLimits final {
  std::uint64_t maxWavefrontSteps;
  std::uint64_t maxEventCount;
  std::uint64_t maxRetainedCaptureBytes;
  std::chrono::steady_clock::duration maxSimulationWallTime =
      std::chrono::steady_clock::duration::max();
};

/// Capture every activation of the finalized Structured candidate's selected
/// Spatial boundary under the common source workload, replay the mechanically
/// derived Canonical Dataflow graph, and compare value, stream, memory, and
/// retirement semantics. The candidate's immutable thread/Spatial ABI is the
/// sole activation-boundary authority. A production semantic gate may provide
/// its already computed source observation so the same execution also proves
/// the whole-program transformation.
llvm::Expected<SourceBackedDfgValidationResult> validateSourceBackedDfgReplay(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    SourceBackedDfgValidationLimits limits,
    const NativeStructuredProgramObservations *sourceObservations = nullptr,
    SourceBackedDfgReplayCasePublisher publishReplayCase = {});

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SOURCEBACKEDDFGVALIDATION_H

#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::frontend {
struct MaterializedOwnershipCandidate;
struct SpatialOwnershipDecisionPoint;
struct SpatialOwnershipScope;
class StructuredProgramCandidate;
} // namespace loom::frontend

namespace loom::sim {
class CanonicalSimulationRuntimeInput;
class CanonicalSimulationWorkload;
struct NativeStructuredProgramObservations;
} // namespace loom::sim

namespace loom::evaluation::models {

struct PreparedStructuredProgramFunctionalEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
  FindingRequestOrdinal functionalMismatchRequest;
};

/// Exact invocation-local inputs needed to prove the selected Structured
/// candidate and every dynamically executed graph activation against the one
/// source workload. These owner views prime a removable provider cache; only
/// the resulting EvaluationEvidence is persistent.
struct StructuredProgramFunctionalReplayInvocation final {
  const ::loom::ArtifactRootReference &workload;
  const ::loom::ArtifactRootReference &runtimeInput;
  const ::loom::frontend::StructuredProgramCandidate &sourceProgram;
  const ::loom::frontend::SpatialOwnershipScope &scope;
  const ::loom::frontend::SpatialOwnershipDecisionPoint &decision;
  const ::loom::frontend::MaterializedOwnershipCandidate &candidate;
  const ::loom::sim::CanonicalSimulationWorkload &simulationWorkload;
  const ::loom::sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput;
  const ::loom::sim::NativeStructuredProgramObservations &sourceObservations;
  ::loom::sim::SourceBackedDfgValidationLimits limits;
};

/// Registers the exact source-versus-selected Structured functional model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredProgramFunctionalModel();

/// Execute and cache the exact source-backed DFG replay for one finalized
/// ownership candidate. Repeated priming for the same candidate/workload pair
/// must produce an identical result.
llvm::Error primeStructuredProgramFunctionalReplay(
    const ::loom::ArtifactRootReference &candidate,
    const StructuredProgramFunctionalReplayInvocation &invocation,
    const ::loom::ArtifactStore &artifactStore);

/// Constructs the finding-only request comparing one exact Structured
/// candidate with the source program owned by the exact workload/runtime pair.
llvm::Expected<PreparedStructuredProgramFunctionalEvaluation>
prepareStructuredProgramFunctionalEvaluation(
    const ::loom::ArtifactRootReference &candidate,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H

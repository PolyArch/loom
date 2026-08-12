#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
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
  const ::loom::frontend::StructuredProgramCandidate &generationParent;
  const ::loom::frontend::StructuredProgramCandidate &sourceProgram;
  const ::loom::frontend::SpatialOwnershipScope &scope;
  const ::loom::frontend::SpatialOwnershipDecisionPoint &decision;
  llvm::ArrayRef<::loom::frontend::StructuredExecutionShapeDecision>
      executionShapeDecisions;
  const ::loom::frontend::MaterializedOwnershipCandidate &candidate;
  const ::loom::sim::CanonicalSimulationWorkload &simulationWorkload;
  const ::loom::sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput;
  const ::loom::sim::NativeStructuredProgramObservations &sourceObservations;
  ::loom::sim::SourceBackedDfgValidationLimits limits;
};

/// Registers the exact source-versus-selected Structured functional model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredProgramFunctionalModel();

EvaluationModelDescriptorRef structuredProgramFunctionalModelDescriptorRef();
CaseSubjectRoleRef structuredProgramFunctionalCandidateRole();

/// Prime the invocation-local source observation shared by functional
/// comparisons for one exact source/workload/runtime tuple. A conflicting
/// second value is rejected as nondeterministic source execution.
llvm::Error primeStructuredProgramSourceObservations(
    const ::loom::ArtifactRootReference &source,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::sim::NativeStructuredProgramObservations &observations);

/// Execute and cache the exact source-backed DFG replay for one finalized
/// ownership candidate. Repeated priming for the same candidate/workload pair
/// must produce an identical result.
llvm::Error primeStructuredProgramFunctionalReplay(
    const ::loom::ArtifactRootReference &candidate,
    const StructuredProgramFunctionalReplayInvocation &invocation,
    const ::loom::ArtifactStore &artifactStore);

/// Return the provider-owned transient replay projection already used to
/// produce functional Evidence for this exact case. This does not execute the
/// candidate again and does not become persistent candidate state.
llvm::Expected<::loom::sim::SourceBackedDfgValidationResult>
getPrimedStructuredProgramFunctionalReplay(
    const ::loom::ArtifactRootReference &candidate,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput);

/// Constructs the finding-only request comparing one exact Structured
/// candidate with the source program owned by the exact workload/runtime pair.
llvm::Expected<PreparedStructuredProgramFunctionalEvaluation>
prepareStructuredProgramFunctionalEvaluation(
    const ::loom::ArtifactRootReference &candidate,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore,
    const ::loom::BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H

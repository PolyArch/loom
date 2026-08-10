#ifndef LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFUNCTIONAL_H
#define LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFUNCTIONAL_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedCanonicalDataflowFunctionalEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
  FindingRequestOrdinal functionalMismatchRequest;
};

/// Invocation-local source lineage used to validate one immutable Dataflow
/// candidate. The exact Structured parent remains an Evaluation case subject;
/// this projection is removable provider state and never becomes identity.
struct CanonicalDataflowFunctionalReplayInvocation final {
  const ArtifactRootReference &workload;
  const ArtifactRootReference &runtimeInput;
  const frontend::StructuredProgramCandidate &sourceProgram;
  const frontend::SpatialOwnershipScope &scope;
  const frontend::SpatialOwnershipDecisionPoint &decision;
  llvm::ArrayRef<frontend::StructuredExecutionShapeDecision>
      executionShapeDecisions;
  const frontend::MaterializedOwnershipCandidate &candidate;
  const sim::CanonicalSimulationWorkload &simulationWorkload;
  const sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput;
  const sim::NativeStructuredProgramObservations &sourceObservations;
  sim::SourceBackedDfgValidationLimits limits;
};

llvm::Error registerCanonicalDataflowFunctionalModel();

EvaluationModelDescriptorRef canonicalDataflowFunctionalModelDescriptorRef();
CaseSubjectRoleRef canonicalDataflowFunctionalCandidateRole();
CaseSubjectRoleRef canonicalDataflowFunctionalStructuredParentRole();

llvm::Error primeCanonicalDataflowFunctionalReplay(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const CanonicalDataflowFunctionalReplayInvocation &invocation,
    const ArtifactStore &artifactStore);

/// Records an already verified replay under the exact Dataflow/Structured
/// pair. The caller must have proved that the Dataflow candidate is the
/// parent's unchanged mechanical D0 projection; changed D* identities must
/// use primeCanonicalDataflowFunctionalReplay and execute independently.
llvm::Error primeCanonicalDataflowFunctionalReplayResult(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const sim::SourceBackedDfgValidationResult &replay);

llvm::Expected<sim::SourceBackedDfgValidationResult>
getPrimedCanonicalDataflowFunctionalReplay(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFunctionalEvaluationCase(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ArtifactStore &artifactStore);

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFunctionalEvaluationCases(
    llvm::ArrayRef<ArtifactRootReference> candidates,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ArtifactStore &artifactStore);

llvm::Expected<PreparedCanonicalDataflowFunctionalEvaluation>
prepareCanonicalDataflowFunctionalEvaluation(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFUNCTIONAL_H

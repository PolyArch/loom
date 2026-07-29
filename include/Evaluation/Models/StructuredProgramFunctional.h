#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDPROGRAMFUNCTIONAL_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedStructuredProgramFunctionalEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
  FindingRequestOrdinal functionalMismatchRequest;
};

/// Registers the exact source-versus-selected Structured functional model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredProgramFunctionalModel();

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

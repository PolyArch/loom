#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace dataflow {
class CanonicalDataflowArtifact;
}

namespace loom::fabric {
class FinalizedFabricRoot;
}

namespace loom::frontend {
class StructuredProgramCandidate;
}

namespace loom::evaluation::models {

struct PreparedStructuredFabricEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

/// Registers the exact low-fidelity StructuredProgram/Fabric analytic model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredFabricAnalyticModel();

/// Constructs and publishes the complete low-confidence metric request for the
/// exact S/F pair. The returned resolution is the complete transient case
/// closure used to execute and validate the resulting Evidence.
llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ::loom::ArtifactRootReference &structuredProgram,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

/// Primes the removable model-result cache from already finalized owner views.
/// The full provider remains the oracle on a miss; this function only avoids
/// re-importing and mechanically re-lowering an exact candidate that the
/// caller has just finalized.
llvm::Error primeStructuredFabricAnalyticResult(
    const ::loom::ArtifactRootReference &structuredProgramReference,
    const ::loom::frontend::StructuredProgramCandidate &structuredProgram,
    const ::dataflow::CanonicalDataflowArtifact &canonicalDataflow,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H

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

namespace loom::evaluation::models {

struct PreparedStructuredFabricEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
};

/// Registers the exact low-fidelity StructuredProgram/Fabric analytic model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredFabricAnalyticModel();

/// Constructs and publishes one Runtime request for the exact S/F pair. The
/// returned resolution is the complete transient case closure used to execute
/// and validate the resulting Evidence.
llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricRuntimeEvaluation(
    const ::loom::ArtifactRootReference &structuredProgram,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H

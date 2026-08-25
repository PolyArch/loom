#ifndef LOOM_EVALUATION_MODELS_SIMULATIONCOMPARISON_H
#define LOOM_EVALUATION_MODELS_SIMULATIONCOMPARISON_H

#include "Evaluation/Evidence.h"

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedSimulationComparisonEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  FindingRequestOrdinal functionalMismatchFinding;
};

llvm::Error registerSimulationComparisonModel();

llvm::Expected<CaseArtifactResolution> resolveSimulationComparisonCase(
    const ArtifactRootReference &referenceExecution,
    const CaseArtifactResolution &referenceResolution,
    const ArtifactRootReference &candidateExecution,
    const CaseArtifactResolution &candidateResolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

llvm::Expected<PreparedSimulationComparisonEvaluation>
prepareSimulationComparisonEvaluation(
    const ArtifactRootReference &referenceExecution,
    const CaseArtifactResolution &referenceResolution,
    const ArtifactRootReference &candidateExecution,
    const CaseArtifactResolution &candidateResolution,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore);

llvm::Expected<EvaluationEvidence> evaluateSimulationComparison(
    const PreparedSimulationComparisonEvaluation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_SIMULATIONCOMPARISON_H

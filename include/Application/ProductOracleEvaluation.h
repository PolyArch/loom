#ifndef LOOM_APPLICATION_PRODUCTORACLEEVALUATION_H
#define LOOM_APPLICATION_PRODUCTORACLEEVALUATION_H

#include "Application/RuntimeManifest.h"
#include "Evaluation/Evidence.h"

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::application {

struct PreparedProductOracleEvaluation final {
  evaluation::EvaluationRequest request;
  evaluation::CaseArtifactResolution resolution;
  evaluation::FindingRequestOrdinal functionalMismatchFinding;
};

llvm::Error registerProductOracleEvaluationModel();

llvm::Expected<PreparedProductOracleEvaluation>
prepareProductOracleEvaluation(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ArtifactRootReference &execution,
    const evaluation::CaseArtifactResolution &executionResolution,
    const ResolvedConfig &config, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_PRODUCTORACLEEVALUATION_H

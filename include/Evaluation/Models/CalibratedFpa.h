#ifndef LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H
#define LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedCanonicalDataflowFabricCalibratedFpaEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
};

llvm::Error registerCalibratedFpaProviders();

/// Constructs and publishes one in-process calibrated FPA Request for an exact
/// Canonical Dataflow/Fabric pair and immutable FPA parameter bundle. The
/// returned Request can only select the registered in-process provider form.
llvm::Expected<PreparedCanonicalDataflowFabricCalibratedFpaEvaluation>
prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
    const ArtifactRootReference &canonicalDataflow,
    const ArtifactRootReference &fabric, const EdaPredictionModelWeight &weight,
    llvm::ArrayRef<EvaluationCondition> operatingConditions,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H

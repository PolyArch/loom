#ifndef LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H
#define LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H

#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedFabricLowConfidenceEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

llvm::Error registerFabricLowConfidenceModel();

EvaluationModelDescriptorRef fabricLowConfidenceModelDescriptorRef();
CaseSubjectRoleRef fabricHardwareAnalysisSubjectRole();

llvm::Expected<PreparedFabricLowConfidenceEvaluation>
prepareFabricLowConfidenceEvaluation(
    const ArtifactRootReference &fabric,
    llvm::ArrayRef<EvaluationCondition> conditions,
    llvm::ArrayRef<MetricKind> metrics, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_FABRICLOWCONFIDENCE_H

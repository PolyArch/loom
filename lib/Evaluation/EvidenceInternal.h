#ifndef LOOM_LIB_EVALUATION_EVIDENCEINTERNAL_H
#define LOOM_LIB_EVALUATION_EVIDENCEINTERNAL_H

#include "Evaluation/Evidence.h"

namespace loom::evaluation::detail {

class EvaluationEvidenceBuilder final {
public:
  static llvm::Expected<EvaluationEvidence> getForVerifiedRequest(
      const EvaluationRequest &request,
      std::vector<ModelOutputBinding> outputBindings,
      EvaluationEvidenceOutcome outcome,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
};

} // namespace loom::evaluation::detail

#endif // LOOM_LIB_EVALUATION_EVIDENCEINTERNAL_H

#ifndef LOOM_EVALUATION_MODELPROVIDER_H
#define LOOM_EVALUATION_MODELPROVIDER_H

#include "Evaluation/Evidence.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::evaluation {

/// One provider-produced result before the Evaluation owner finalizes the
/// persistent Evidence root. Normalized observations remain in the ordinary
/// Evidence outcome; descriptor outputs remain in their typed output slots.
struct EvaluationModelResult final {
  std::vector<ModelOutputBinding> outputBindings;
  EvaluationEvidenceOutcome outcome;
};

/// Runtime availability for one exact static model descriptor. The descriptor
/// owns model semantics and accepted Requests; this record owns only the
/// executable implementation currently available in this process.
struct EvaluationModelProvider final {
  EvaluationModelDescriptorRef descriptor;
  llvm::Expected<EvaluationModelResult> (*evaluate)(
      const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);
};

/// Registers one provider with static storage duration. Re-registering the
/// same record is a no-op. A second provider for the same exact descriptor is
/// rejected rather than becoming an ambient selection mechanism.
llvm::Error
registerEvaluationModelProvider(const EvaluationModelProvider &provider);

const EvaluationModelProvider *
findEvaluationModelProvider(EvaluationModelDescriptorRef descriptor);

/// Verifies and executes one exact Request. Provider absence is a stable
/// Unsupported outcome for this exact Request. Provider-produced output and
/// observation cardinality is validated by the sole Evidence constructor.
llvm::Expected<EvaluationEvidence>
evaluateRequest(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifactStore);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELPROVIDER_H

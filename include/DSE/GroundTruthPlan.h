#ifndef LOOM_DSE_GROUNDTRUTHPLAN_H
#define LOOM_DSE_GROUNDTRUTHPLAN_H

#include "Config/ResolvedConfig.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {

struct GroundTruthEvidencePartitions final {
  std::vector<ArtifactRootReference> training;
  std::vector<ArtifactRootReference> validation;
  std::vector<ArtifactRootReference> heldOut;
  std::optional<ArtifactRootReference> priorParameterBundle;
};

struct GroundTruthModelTrack final {
  GroundTruthEvidencePartitions evidence;
  DeterministicGbdtTrainingConfig training;
  evaluation::ExactRatio calibrationQuantile;
  evaluation::DecimalValue maximumValidationError;
  evaluation::DecimalValue maximumHeldOutError;
};

struct GroundTruthPlanInputs final {
  std::optional<GroundTruthModelTrack> fpa;
  std::optional<GroundTruthModelTrack> systemRuntime;
};

struct GroundTruthTrackOutputs final {
  PlanOutputRef trainedBundle;
  PlanOutputRef validationEvidence;
  PlanOutputRef releasedBundle;
  PlanOutputRef heldOutEvidence;
};

class ResolvedGroundTruthPlan final {
public:
  const ResolvedConfig &resolvedConfig() const { return resolvedConfig_; }
  const ResolvedDseConfigView &view() const { return view_; }
  llvm::ArrayRef<ArtifactRootReference> semanticInputs() const {
    return semanticInputs_;
  }
  llvm::ArrayRef<ArtifactRootReference> preexistingEvidence() const {
    return preexistingEvidence_;
  }
  const std::optional<GroundTruthTrackOutputs> &fpaOutputs() const {
    return fpaOutputs_;
  }
  const std::optional<GroundTruthTrackOutputs> &systemRuntimeOutputs() const {
    return systemRuntimeOutputs_;
  }

private:
  ResolvedGroundTruthPlan(
      ResolvedConfig resolvedConfig, ResolvedDseConfigView view,
      std::vector<ArtifactRootReference> semanticInputs,
      std::vector<ArtifactRootReference> preexistingEvidence,
      std::optional<GroundTruthTrackOutputs> fpaOutputs,
      std::optional<GroundTruthTrackOutputs> systemRuntimeOutputs)
      : resolvedConfig_(std::move(resolvedConfig)), view_(std::move(view)),
        semanticInputs_(std::move(semanticInputs)),
        preexistingEvidence_(std::move(preexistingEvidence)),
        fpaOutputs_(std::move(fpaOutputs)),
        systemRuntimeOutputs_(std::move(systemRuntimeOutputs)) {}

  ResolvedConfig resolvedConfig_;
  ResolvedDseConfigView view_;
  std::vector<ArtifactRootReference> semanticInputs_;
  std::vector<ArtifactRootReference> preexistingEvidence_;
  std::optional<GroundTruthTrackOutputs> fpaOutputs_;
  std::optional<GroundTruthTrackOutputs> systemRuntimeOutputs_;

  friend llvm::Expected<ResolvedGroundTruthPlan>
      buildGroundTruthPlan(ResolvedConfig, GroundTruthPlanInputs);
};

llvm::Expected<ResolvedGroundTruthPlan>
buildGroundTruthPlan(ResolvedConfig baseConfig, GroundTruthPlanInputs inputs);

} // namespace loom::dse

#endif // LOOM_DSE_GROUNDTRUTHPLAN_H

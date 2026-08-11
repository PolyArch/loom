#ifndef LOOM_DSE_GROUNDTRUTHPLAN_H
#define LOOM_DSE_GROUNDTRUTHPLAN_H

#include "Config/ResolvedConfig.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

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

/// Exact routed HardwareImplementation members assigned to one calibration
/// partition. The roots remain ordinary Hardware Artifacts; this record owns
/// only the finite plan input set.
struct FpaGroundTruthPartitionInputs final {
  std::vector<ArtifactRootReference> hardwareImplementations;
};

struct FpaGroundTruthPlanInputs final {
  FpaGroundTruthPartitionInputs training;
  FpaGroundTruthPartitionInputs validation;
  FpaGroundTruthPartitionInputs heldOut;
  std::vector<evaluation::EvaluationCondition> operatingConditions;
};

struct FpaGroundTruthCollectionPlan final {
  ResolvedConfig resolvedConfig;
  PlanOutputRef trainingEvidence;
  PlanOutputRef validationEvidence;
  PlanOutputRef heldOutEvidence;
};

llvm::Expected<FpaGroundTruthCollectionPlan> buildFpaGroundTruthCollectionPlan(
    FpaGroundTruthPlanInputs inputs, const ResolvedConfig &baseConfig,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::dse

#endif // LOOM_DSE_GROUNDTRUTHPLAN_H

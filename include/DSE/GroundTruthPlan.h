#ifndef LOOM_DSE_GROUNDTRUTHPLAN_H
#define LOOM_DSE_GROUNDTRUTHPLAN_H

#include "Config/ResolvedConfig.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

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

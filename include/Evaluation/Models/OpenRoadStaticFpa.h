#ifndef LOOM_EVALUATION_MODELS_OPENROADSTATICFPA_H
#define LOOM_EVALUATION_MODELS_OPENROADSTATICFPA_H

#include "Evaluation/Models/OpenRoadStaticFpaConfig.h"
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct CompleteOpenRoadStaticFpaConfiguration final {
  OpenRoadStaticFpaProviderBinding providerBinding;
  ProcessCornerCondition processCorner;
  SupplyVoltageCondition supplyVoltage;
  TemperatureCondition temperature;
  RequiredClockPeriodCondition clockPeriod;
  std::optional<ActivityBindingCondition> activity;
  std::optional<ExplicitAssumptionSource> activityAssumption;
  std::vector<MetricKind> metrics;
};

struct PreparedOpenRoadStaticFpaEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

llvm::Error registerOpenRoadStaticFpaModel();

EvaluationModelDescriptorRef openRoadStaticFpaModelDescriptorRef();

const ResolvedModelConfigViewContract &openRoadStaticFpaConfigViewContract();

llvm::Expected<PreparedOpenRoadStaticFpaEvaluation>
prepareOpenRoadStaticFpaEvaluation(
    const ArtifactRootReference &hardwareImplementation,
    llvm::ArrayRef<EvaluationCondition> conditions,
    llvm::ArrayRef<MetricKind> metrics, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

llvm::Expected<CompleteOpenRoadStaticFpaConfiguration>
projectCompleteOpenRoadStaticFpaConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_OPENROADSTATICFPA_H

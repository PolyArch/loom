#ifndef LOOM_MODELING_ANALYTICMODELSUPPORT_H
#define LOOM_MODELING_ANALYTICMODELSUPPORT_H

#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom {
class ArtifactStore;
}

namespace dataflow {
class CanonicalDataflowProgramView;
}

namespace loom::fabric {
class FinalizedFabricRoot;
}

namespace loom::evaluation::models::detail {

const ResolvedModelConfigViewContract &emptyLowConfidenceConfigView();

struct AnalyticWorkloadEstimate final {
  std::uint64_t schedulingPressure = 0;
  std::uint64_t activityUnits = 0;
};

struct LowConfidenceMetricSet final {
  std::uint64_t runtimePicoseconds = 0;
  std::uint64_t limitingClockFrequencyHertz = 0;
  std::uint64_t totalAreaSquareMicrometers = 0;
  std::uint64_t dynamicPowerMicrowatts = 0;
  std::uint64_t leakagePowerMicrowatts = 0;

  llvm::Expected<MetricResult> result(MetricKind metric) const;
};

llvm::Expected<CaseArtifactResolution>
resolveSingleSubjectFabricCase(const ArtifactRootReference &subject,
                               const ArtifactRootReference &fabric,
                               const ArtifactStore &artifactStore);

llvm::Expected<LowConfidenceMetricSet>
estimateLowConfidenceMetrics(std::uint64_t instructionLeaves,
                             AnalyticWorkloadEstimate workload,
                             const fabric::FinalizedFabricRoot &fabricRoot);

llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowWorkload(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot);

} // namespace loom::evaluation::models::detail

#endif // LOOM_MODELING_ANALYTICMODELSUPPORT_H

#ifndef LOOM_MODELING_ANALYTICMODELSUPPORT_H
#define LOOM_MODELING_ANALYTICMODELSUPPORT_H

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"

#include "llvm/ADT/ArrayRef.h"
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
  std::uint64_t graphActivations = 0;
  std::uint64_t boundaryPayloadBytes = 0;
  std::uint64_t memoryBoundaryBindings = 0;
  std::uint64_t memoryTransactions = 0;
};

struct LowConfidencePhysicalActivity final {
  ExactRatio staticProbability;
  ExactRatio transitionsPerClock;
};

struct LowConfidenceMetricSet final {
  std::uint64_t runtimePicoseconds = 0;
  std::uint64_t limitingClockFrequencyHertz = 0;
  std::uint64_t totalAreaSquareMicrometers = 0;
  std::uint64_t dynamicPowerMicrowatts = 0;
  std::uint64_t leakagePowerMicrowatts = 0;

  friend bool operator==(const LowConfidenceMetricSet &lhs,
                         const LowConfidenceMetricSet &rhs) {
    return lhs.runtimePicoseconds == rhs.runtimePicoseconds &&
           lhs.limitingClockFrequencyHertz == rhs.limitingClockFrequencyHertz &&
           lhs.totalAreaSquareMicrometers == rhs.totalAreaSquareMicrometers &&
           lhs.dynamicPowerMicrowatts == rhs.dynamicPowerMicrowatts &&
           lhs.leakagePowerMicrowatts == rhs.leakagePowerMicrowatts;
  }
  friend bool operator!=(const LowConfidenceMetricSet &lhs,
                         const LowConfidenceMetricSet &rhs) {
    return !(lhs == rhs);
  }

  llvm::Expected<MetricResult> result(MetricKind metric) const;
};

llvm::Expected<std::int64_t>
lowConfidenceMetricQuantumBase10Exponent(MetricKind metric);

llvm::Expected<CaseArtifactResolution> resolveSingleSubjectFabricCase(
    const ArtifactRootReference &subject, const ArtifactRootReference &fabric,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<CaseArtifactResolution::Entry> additionalEntries = {});

llvm::Expected<LowConfidenceMetricSet>
estimateLowConfidenceMetrics(std::uint64_t instructionLeaves,
                             AnalyticWorkloadEstimate workload,
                             const fabric::FinalizedFabricRoot &fabricRoot);

/// Projects the hardware-only portion of the same fixed low-confidence model.
/// Dynamic power is meaningful only when the caller supplies one exact typed
/// activity assumption; the absent branch contains no hidden toggle default.
llvm::Expected<LowConfidenceMetricSet> estimateLowConfidencePhysicalMetrics(
    const fabric::FinalizedFabricRoot &fabricRoot,
    std::optional<LowConfidencePhysicalActivity> activity);

llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowWorkload(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot);

/// Projects one exact graph definition from the same canonical owner. This is
/// used when a complete Structured candidate contains graphs with different
/// dynamic activation counts.
llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowGraphWorkload(
    const ::dataflow::CanonicalDataflowProgramView &program,
    ::dataflow::GraphRef graph, const fabric::FinalizedFabricRoot &fabricRoot);

} // namespace loom::evaluation::models::detail

#endif // LOOM_MODELING_ANALYTICMODELSUPPORT_H

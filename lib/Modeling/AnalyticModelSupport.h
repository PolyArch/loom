#ifndef LOOM_MODELING_ANALYTICMODELSUPPORT_H
#define LOOM_MODELING_ANALYTICMODELSUPPORT_H

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Models/SystemRuntimeAnalytic.h"

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

/// Static per-firing shape of one Canonical Dataflow graph (or program)
/// against one Fabric. Scheduling pressure is the resource-bound initiation
/// interval, the critical path and recurrence lengths bound one firing's
/// latency, and the external memory bytes are what one firing moves across
/// the SpatialCore memory boundary. Activity units and the remaining counts
/// feed the physical (power) estimate only.
struct AnalyticWorkloadEstimate final {
  std::uint64_t schedulingPressure = 0;
  std::uint64_t criticalPathLength = 0;
  std::uint64_t recurrenceLength = 0;
  std::uint64_t externalMemoryBytes = 0;
  std::uint64_t activityUnits = 0;
  std::uint64_t graphActivations = 0;
  std::uint64_t boundaryPayloadBytes = 0;
  std::uint64_t memoryBoundaryBindings = 0;
  std::uint64_t memoryTransactions = 0;
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

/// Whole-program low-confidence metrics: the serialized host residual plus
/// every launch site's roofline duration under the widest useful AccCore
/// allocation, with physical activity from the accumulated workload.
llvm::Expected<LowConfidenceMetricSet>
estimateLowConfidenceMetrics(std::uint64_t instructionLeaves,
                             AnalyticWorkloadEstimate workload,
                             llvm::ArrayRef<AnalyticLaunchEstimate> launches,
                             const SystemPlatformModel &platform,
                             const fabric::FinalizedFabricRoot &fabricRoot);

/// The platform model of the exact System root, or absent when the Fabric is
/// not a complete System (the runtime model is inapplicable, not zero).
llvm::Expected<std::optional<SystemPlatformModel>>
projectFabricPlatformModel(const fabric::FinalizedFabricRoot &fabricRoot);

/// Estimates hardware-only physical metrics from the same complete Fabric
/// inventory and coefficient table used by the software-aware models.
/// `activityPartsPer1024` is a descriptor-owned projection of explicit
/// activity assumptions; callers must not supply an implicit default.
llvm::Expected<LowConfidenceMetricSet> estimateLowConfidenceFabricMetrics(
    const fabric::FinalizedFabricRoot &fabricRoot,
    std::uint64_t activityPartsPer1024);

/// The clock basis shared by every low-confidence estimate over one Fabric:
/// its structure-derived critical delay. The analytic models report its
/// reciprocal as LimitingClockFrequency, so a consumer that expresses measured
/// cycles in the analytic picosecond domain multiplies by exactly this period.
llvm::Expected<std::uint64_t>
lowConfidenceClockPeriodPicoseconds(const fabric::FinalizedFabricRoot &fabricRoot);

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

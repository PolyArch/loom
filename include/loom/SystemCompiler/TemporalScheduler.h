//===-- TemporalScheduler.h - Multi-mode temporal scheduling --------*- C++ -*-===//
//
// Temporal scheduling for the Tapestry multi-core compiler with four modes:
// BATCH_SEQUENTIAL, PIPELINE_PARALLEL, SPATIAL_PARALLEL, and SPATIAL_SHARING.
//
// BATCH_SEQUENTIAL: sequential kernel execution with reconfiguration gaps.
// PIPELINE_PARALLEL: pipelined execution with overlap across cores.
// SPATIAL_PARALLEL: all cores start simultaneously, sequential within each.
// SPATIAL_SHARING: co-located kernels share the PE array concurrently.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_TEMPORALSCHEDULER_H
#define LOOM_SYSTEMCOMPILER_TEMPORALSCHEDULER_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/CostSummary.h"
#include "loom/SystemCompiler/ExecutionModel.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <string>
#include <vector>

namespace loom {

/// Multi-mode temporal scheduler for the Tapestry system compiler.
///
/// Dispatches to mode-specific scheduling algorithms based on the
/// ExecutionModelConfig. Produces a TemporalSchedule describing the
/// time-domain behavior of all kernels across all cores.
class TemporalScheduler {
public:
  /// Compute a temporal schedule for the given assignment and kernel metrics.
  ///
  /// \param assignment     L1 core assignment result.
  /// \param costSummaries  Per-core L2 compilation metrics.
  /// \param contracts      Inter-kernel communication contracts.
  /// \param config         Execution model configuration (mode, reconfigCycles).
  /// \param[out] outSchedule  The computed temporal schedule.
  /// \returns Error string on failure, empty string on success.
  std::string schedule(const AssignmentResult &assignment,
                       const std::vector<CoreCostSummary> &costSummaries,
                       const std::vector<ContractSpec> &contracts,
                       const ExecutionModelConfig &config,
                       TemporalSchedule &outSchedule);

private:
  /// PIPELINE_PARALLEL: ASAP scheduling with pipeline overlap.
  /// Computes start times that respect dependency ordering and same-core
  /// serialization. Kernels on different cores can overlap if the consumer
  /// starts after the producer emits its first tile.
  std::string computePipelineSchedule(
      const AssignmentResult &assignment,
      const std::vector<CoreCostSummary> &costSummaries,
      const std::vector<ContractSpec> &contracts,
      const ExecutionModelConfig &config,
      TemporalSchedule &outSchedule);

  /// SPATIAL_PARALLEL: all cores start at time 0, sequential within cores.
  /// System latency = max(core latency) + NoC overhead.
  std::string computeSpatialParallelSchedule(
      const AssignmentResult &assignment,
      const std::vector<CoreCostSummary> &costSummaries,
      const std::vector<ContractSpec> &contracts,
      const ExecutionModelConfig &config,
      TemporalSchedule &outSchedule);

  /// SPATIAL_SHARING: co-located kernels execute concurrently via spatial
  /// partitioning. Per-core latency = max(kernel durations) with no reconfig.
  std::string computeSpatialSharingSchedule(
      const AssignmentResult &assignment,
      const std::vector<CoreCostSummary> &costSummaries,
      const std::vector<ContractSpec> &contracts,
      const ExecutionModelConfig &config,
      TemporalSchedule &outSchedule);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TEMPORALSCHEDULER_H

#include "SpatialPnrExecution.h"

#include "Common/MappingDebugLog.h"
#include "PnR/SpatialPnrProblem.h"

#include <algorithm>
#include <limits>

namespace loom::pnr::detail {

SpatialPnrWorkerAllocation
resolveWorkerAllocation(std::uint32_t configuredWorkerCount,
                        std::uint32_t restartCount,
                        const SpatialActiveProblemStatistics &problemStatistics,
                        std::uint64_t workerScratchReservationBytes,
                        ExecutionResourceBudget executionBudget,
                        bool memoryCalibrated) {
  SpatialPnrWorkerAllocation allocation;
  allocation.configuredWorkerCount = configuredWorkerCount;
  allocation.restartCount = restartCount;
  allocation.sharedProblemRetainedBytes =
      problemStatistics.context.retainedBytes;
  const auto saturatingAdd = [](std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
               ? std::numeric_limits<std::uint64_t>::max()
               : lhs + rhs;
  };
  allocation.activeRouteGraphUnitCount =
      saturatingAdd(saturatingAdd(problemStatistics.activeEndpointCount,
                                  problemStatistics.activeTraversalCount),
                    problemStatistics.activeRoutingArcCount);
  allocation.activeRouteGraphUnitCount =
      std::max(UINT64_C(1), allocation.activeRouteGraphUnitCount);
  allocation.routeGraphLimitedWorkerCount = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(allocation.activeRouteGraphUnitCount,
                              std::numeric_limits<std::uint32_t>::max()));
  allocation.workerScratchReservationBytes = workerScratchReservationBytes;
  allocation.actualWorkerCount =
      std::min({configuredWorkerCount, restartCount,
                allocation.routeGraphLimitedWorkerCount});
  if (executionBudget.cpuCores) {
    allocation.cpuLimitedWorkerCount = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(*executionBudget.cpuCores,
                                std::numeric_limits<std::uint32_t>::max()));
    allocation.actualWorkerCount = std::min(allocation.actualWorkerCount,
                                            *allocation.cpuLimitedWorkerCount);
  }
  if (executionBudget.memoryBytes) {
    const std::uint64_t workerBytes =
        *executionBudget.memoryBytes > allocation.sharedProblemRetainedBytes
            ? *executionBudget.memoryBytes -
                  allocation.sharedProblemRetainedBytes
            : 0;
    const std::uint64_t memoryWorkers =
        allocation.workerScratchReservationBytes == 0
            ? 1
            : std::max(UINT64_C(1),
                       workerBytes / allocation.workerScratchReservationBytes);
    allocation.memoryLimitedWorkerCount =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            memoryWorkers, std::numeric_limits<std::uint32_t>::max()));
    allocation.actualWorkerCount = std::min(
        allocation.actualWorkerCount, *allocation.memoryLimitedWorkerCount);
  }
  allocation.memoryCalibrated = memoryCalibrated;
  return allocation;
}

void emitInvocationExecutionStatistics(
    const SpatialPnrWorkerAllocation &allocation,
    const SpatialActiveProblemStatistics &problemStatistics,
    ExecutionResourceBudget executionBudget,
    const ExecutionResourceTracker &resources, bool preparedSeedHandoff,
    bool borrowedInvocationPool) {
  if (!mapping_debug::enabled(mapping_debug::Level::Summary))
    return;
  const ExecutionResourceStatistics observation = resources.observe();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "spatial_pnr_execution";
        fields["prepared_seed_handoff"] = preparedSeedHandoff;
        fields["configured_worker_count"] = allocation.configuredWorkerCount;
        fields["restart_count"] = allocation.restartCount;
        fields["serial_prefix"] = allocation.serialPrefix;
        fields["worker_count"] = allocation.actualWorkerCount;
        fields["thread_count"] = allocation.actualWorkerCount;
        fields["active_route_graph_unit_count"] =
            allocation.activeRouteGraphUnitCount;
        fields["route_graph_limited_worker_count"] =
            allocation.routeGraphLimitedWorkerCount;
        fields["worker_scratch_reservation_bytes"] =
            allocation.workerScratchReservationBytes;
        fields["maximum_observed_worker_scratch_bytes"] =
            allocation.maximumObservedWorkerScratchBytes;
        fields["shared_problem_retained_bytes"] =
            allocation.sharedProblemRetainedBytes;
        fields["memory_calibrated"] = allocation.memoryCalibrated;
        if (allocation.cpuLimitedWorkerCount)
          fields["cpu_limited_worker_count"] =
              *allocation.cpuLimitedWorkerCount;
        else
          fields["cpu_limited_worker_count"] = nullptr;
        if (allocation.memoryLimitedWorkerCount)
          fields["memory_limited_worker_count"] =
              *allocation.memoryLimitedWorkerCount;
        else
          fields["memory_limited_worker_count"] = nullptr;
        if (executionBudget.memoryBytes)
          fields["memory_budget_bytes"] = *executionBudget.memoryBytes;
        else
          fields["memory_budget_bytes"] = nullptr;
        if (executionBudget.cpuCores)
          fields["cpu_budget_cores"] = *executionBudget.cpuCores;
        else
          fields["cpu_budget_cores"] = nullptr;
        fields["active_endpoint_count"] = problemStatistics.activeEndpointCount;
        fields["active_traversal_count"] =
            problemStatistics.activeTraversalCount;
        fields["active_routing_arc_count"] =
            problemStatistics.activeRoutingArcCount;
        fields["active_wall_time_ns"] = observation.activeWallTimeNanoseconds;
        if (observation.processCpuTimeDeltaNanoseconds)
          fields["process_cpu_time_delta_ns"] =
              *observation.processCpuTimeDeltaNanoseconds;
        else
          fields["process_cpu_time_delta_ns"] = nullptr;
        fields["resource_observation_scope"] = "process";
        fields["overlapping_frontier_process_observation"] =
            borrowedInvocationPool;
        fields["allocated_memory_bytes"] = observation.allocatedMemoryBytes;
        if (observation.peakResidentMemoryBytes)
          fields["peak_resident_memory_bytes"] =
              *observation.peakResidentMemoryBytes;
        else
          fields["peak_resident_memory_bytes"] = nullptr;
      });
}

} // namespace loom::pnr::detail

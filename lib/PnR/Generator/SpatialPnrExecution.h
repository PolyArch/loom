#ifndef LOOM_PNR_GENERATOR_SPATIALPNREXECUTION_H
#define LOOM_PNR_GENERATOR_SPATIALPNREXECUTION_H

#include "Common/ExecutionControl.h"

#include <cstdint>
#include <optional>

namespace loom::pnr {
struct SpatialActiveProblemStatistics;
namespace detail {

struct SpatialPnrWorkerAllocation final {
  std::uint32_t configuredWorkerCount = 1;
  std::uint32_t restartCount = 1;
  std::uint32_t actualWorkerCount = 1;
  std::uint64_t activeRouteGraphUnitCount = 1;
  std::uint64_t workerScratchReservationBytes = 0;
  std::uint64_t maximumObservedWorkerScratchBytes = 0;
  std::uint64_t sharedProblemRetainedBytes = 0;
  std::optional<std::uint32_t> cpuLimitedWorkerCount;
  std::optional<std::uint32_t> memoryLimitedWorkerCount;
  std::uint32_t routeGraphLimitedWorkerCount = 1;
  bool serialPrefix = false;
  bool memoryCalibrated = false;
};

SpatialPnrWorkerAllocation
resolveWorkerAllocation(std::uint32_t configuredWorkerCount,
                        std::uint32_t restartCount,
                        const SpatialActiveProblemStatistics &problemStatistics,
                        std::uint64_t workerScratchReservationBytes,
                        ExecutionResourceBudget executionBudget,
                        bool memoryCalibrated);

void emitInvocationExecutionStatistics(
    const SpatialPnrWorkerAllocation &allocation,
    const SpatialActiveProblemStatistics &problemStatistics,
    ExecutionResourceBudget executionBudget,
    const ExecutionResourceTracker &resources, bool preparedSeedHandoff,
    bool borrowedInvocationPool);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_PNR_GENERATOR_SPATIALPNREXECUTION_H

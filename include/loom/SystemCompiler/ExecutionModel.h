//===-- ExecutionModel.h - Temporal execution model types ----------*- C++ -*-===//
//
// Types and functions for the temporal execution model in the Tapestry
// multi-core compilation pipeline. Defines how kernels are scheduled in time
// across cores: currently supports BATCH_SEQUENTIAL mode where kernels execute
// one at a time with reconfiguration gaps.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_EXECUTIONMODEL_H
#define LOOM_SYSTEMCOMPILER_EXECUTIONMODEL_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/CostSummary.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Execution Mode
//===----------------------------------------------------------------------===//

/// Temporal execution mode for multi-kernel scheduling on cores.
enum class ExecutionMode {
  /// All cores run kernel A, barrier, all run kernel B, etc.
  /// Each kernel occupies the full PE array of its assigned core.
  BATCH_SEQUENTIAL,

  /// X cores for kernel A, (N-X) for kernel B simultaneously.
  /// Not implemented in current version.
  PIPELINE_PARALLEL,

  /// Kernels share PE array within one core concurrently.
  /// Not implemented in current version.
  SPATIAL_SHARING
};

/// Convert ExecutionMode to a human-readable string.
const char *executionModeToString(ExecutionMode mode);

/// Parse an ExecutionMode from a string. Returns BATCH_SEQUENTIAL if unknown.
ExecutionMode executionModeFromString(const std::string &s);

//===----------------------------------------------------------------------===//
// Execution Model Configuration
//===----------------------------------------------------------------------===//

/// Configuration for the temporal execution model.
struct ExecutionModelConfig {
  /// Which temporal execution mode to use.
  ExecutionMode mode = ExecutionMode::BATCH_SEQUENTIAL;

  /// Cost in cycles to switch configuration on a single core.
  /// Applied between sequential kernels on the same core.
  unsigned reconfigCycles = 100;

  /// Global synchronization overhead in cycles (barrier between batches).
  unsigned barrierCycles = 10;
};

//===----------------------------------------------------------------------===//
// Temporal Schedule
//===----------------------------------------------------------------------===//

/// Timing information for a single kernel execution.
struct KernelTiming {
  std::string kernelName;

  /// Number of outermost loop iterations.
  unsigned tripCount = 0;

  /// Cycles per iteration (initiation interval from mapper).
  unsigned achievedII = 0;

  /// Total execution cycles = tripCount * achievedII.
  /// This is the physically meaningful execution time.
  uint64_t executionCycles = 0;
};

/// Schedule for a single core, specifying the order and timing of its kernels.
struct CoreSchedule {
  std::string coreInstanceName;

  /// Kernels in topological execution order.
  std::vector<std::string> kernelOrder;

  /// Timing for each kernel, parallel to kernelOrder.
  std::vector<KernelTiming> kernelTimings;

  /// Total cycles on this core: sum(executionCycles) + reconfig gaps.
  uint64_t totalCycles = 0;

  /// Number of reconfiguration events on this core.
  unsigned reconfigCount = 0;
};

/// Complete temporal schedule for the system.
struct TemporalSchedule {
  /// Per-core schedules.
  std::vector<CoreSchedule> coreSchedules;

  /// Critical-path latency across all cores (max core latency).
  uint64_t maxCoreCycles = 0;

  /// Inter-core NoC transfer overhead (aggregate).
  uint64_t nocOverheadCycles = 0;

  /// System-level latency: maxCoreCycles + nocOverheadCycles.
  uint64_t systemLatencyCycles = 0;

  /// The execution mode used to produce this schedule.
  ExecutionMode mode = ExecutionMode::BATCH_SEQUENTIAL;
};

//===----------------------------------------------------------------------===//
// Temporal Scheduler
//===----------------------------------------------------------------------===//

/// Compute the temporal schedule for a given assignment and kernel metrics.
///
/// In BATCH_SEQUENTIAL mode:
///   - Kernels on each core are ordered by topological sort of the TDG.
///   - Per-kernel execution = tripCount * achievedII.
///   - Reconfig gaps are inserted between consecutive kernels on the same core.
///   - System latency = max core latency + inter-core NoC overhead.
///
/// Returns an error string on failure (empty string on success).
std::string computeTemporalSchedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule);

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Topologically sort kernels assigned to a single core, respecting the
/// dependency edges in the contracts. Returns the sorted kernel names.
/// If no dependency ordering exists, preserves the input order.
std::vector<std::string>
topologicalSortKernels(const std::vector<std::string> &kernels,
                       const std::vector<ContractSpec> &contracts);

/// Look up the trip count for a kernel. Currently uses a default value
/// if no annotation is available. Returns the trip count.
unsigned lookupTripCount(const std::string &kernelName,
                         const std::vector<CoreCostSummary> &costSummaries);

/// Compute NoC transfer overhead in cycles from inter-core contract edges.
uint64_t computeNoCOverhead(const AssignmentResult &assignment,
                            const std::vector<ContractSpec> &contracts);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_EXECUTIONMODEL_H

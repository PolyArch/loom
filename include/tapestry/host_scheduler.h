//===-- host_scheduler.h - Multi-kernel host code generation ------*- C++ -*-===//
//
// Generates host-side scheduling code for multi-kernel TDG execution.
// Handles CGRA config loading, kernel launches, host function calls,
// DMA transfers between host and CGRA, and execution ordering from
// topological sort of the TDG.
//
//===----------------------------------------------------------------------===//

#ifndef TAPESTRY_HOST_SCHEDULER_H
#define TAPESTRY_HOST_SCHEDULER_H

#include "loom/SystemCompiler/SystemTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// Kernel Execution Target
//===----------------------------------------------------------------------===//

enum class ExecutionTarget {
  CGRA, // Runs on a CGRA core
  HOST, // Runs on the host CPU
};

//===----------------------------------------------------------------------===//
// Scheduled Kernel
//===----------------------------------------------------------------------===//

/// A kernel in the scheduled execution order with assignment metadata.
struct ScheduledKernel {
  std::string name;
  ExecutionTarget target = ExecutionTarget::CGRA;

  /// For CGRA kernels: which core instance it maps to.
  int coreInstanceIdx = -1;
  std::string coreTypeName;

  /// For HOST kernels: the C function name to call.
  std::string hostFunctionName;

  /// Estimated execution cycles (for comments/documentation).
  uint64_t estimatedCycles = 0;
};

//===----------------------------------------------------------------------===//
// DMA Transfer Descriptor
//===----------------------------------------------------------------------===//

/// Describes a DMA transfer between host memory and CGRA SPM (or vice versa).
struct HostDMATransfer {
  enum Direction { HOST_TO_CGRA, CGRA_TO_HOST, CGRA_TO_CGRA };
  Direction direction = HOST_TO_CGRA;

  std::string producerKernel;
  std::string consumerKernel;

  /// Source and destination addresses (symbolic names in generated code).
  std::string srcAddrSymbol;
  std::string dstAddrSymbol;

  /// Transfer size in bytes.
  uint64_t transferSizeBytes = 0;

  /// Core instance indices for CGRA endpoints.
  int srcCoreIdx = -1;
  int dstCoreIdx = -1;
};

//===----------------------------------------------------------------------===//
// Synchronization Barrier
//===----------------------------------------------------------------------===//

/// A synchronization point where the host waits for one or more CGRA cores.
struct SyncBarrier {
  std::string label;
  std::vector<unsigned> waitCoreIds;
};

//===----------------------------------------------------------------------===//
// Host Schedule
//===----------------------------------------------------------------------===//

/// Complete host-side execution schedule for a multi-kernel TDG.
struct HostSchedule {
  /// Kernels in topological execution order.
  std::vector<ScheduledKernel> executionOrder;

  /// DMA transfers required between execution stages.
  std::vector<HostDMATransfer> dmaTransfers;

  /// Synchronization barriers.
  std::vector<SyncBarrier> barriers;

  /// Name of the pipeline (used in generated code identifiers).
  std::string pipelineName;
};

//===----------------------------------------------------------------------===//
// Host Schedule Options
//===----------------------------------------------------------------------===//

struct HostScheduleOptions {
  /// Insert comments with timing estimates in generated code.
  bool emitTimingComments = true;

  /// Generate double-buffer management code.
  bool enableDoubleBuffering = false;

  /// Include error-checking code after each DMA/launch call.
  bool emitErrorChecks = true;

  /// Use the runtime API prefix (e.g., "loom_" or "tapestry_").
  std::string runtimePrefix = "tapestry_";

  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Host Scheduler
//===----------------------------------------------------------------------===//

/// Builds and generates the host-side execution schedule.
class HostScheduler {
public:
  explicit HostScheduler(const HostScheduleOptions &options);

  /// Build a HostSchedule from the optimization result.
  ///
  /// \param kernels     Kernel descriptors with assignment metadata.
  /// \param contracts   Inter-kernel contracts defining data flow.
  /// \param assignments Benders result with core assignments.
  /// \param arch        System architecture for core type info.
  /// \returns           The complete host schedule.
  HostSchedule buildSchedule(
      const std::vector<loom::tapestry::KernelDesc> &kernels,
      const std::vector<loom::tapestry::ContractSpec> &contracts,
      const loom::tapestry::BendersResult &assignments,
      const loom::tapestry::SystemArchitecture &arch);

  /// Generate a C source file implementing the host schedule.
  ///
  /// \param schedule    The schedule to generate code for.
  /// \param outputPath  Path for the generated host_driver.c file.
  /// \returns           True on success.
  bool generateHostCode(const HostSchedule &schedule,
                        const std::string &outputPath);

  /// Generate a C header file with declarations for the host driver.
  ///
  /// \param schedule    The schedule to generate declarations for.
  /// \param outputPath  Path for the generated host_driver.h file.
  /// \returns           True on success.
  bool generateHostHeader(const HostSchedule &schedule,
                          const std::string &outputPath);

private:
  /// Topological sort of kernels based on contract edges.
  std::vector<std::string> topologicalSort(
      const std::vector<loom::tapestry::KernelDesc> &kernels,
      const std::vector<loom::tapestry::ContractSpec> &contracts);

  /// Determine DMA transfers needed at each stage boundary.
  std::vector<HostDMATransfer> planDMATransfers(
      const std::vector<ScheduledKernel> &orderedKernels,
      const std::vector<loom::tapestry::ContractSpec> &contracts);

  /// Determine where synchronization barriers are needed.
  std::vector<SyncBarrier> planBarriers(
      const std::vector<ScheduledKernel> &orderedKernels);

  /// Emit a single kernel launch or call.
  void emitKernelAction(std::string &code, const ScheduledKernel &kernel);

  /// Emit a DMA transfer call.
  void emitDMATransfer(std::string &code, const HostDMATransfer &transfer);

  /// Emit a barrier/wait call.
  void emitBarrier(std::string &code, const SyncBarrier &barrier);

  HostScheduleOptions options_;
};

} // namespace tapestry

#endif // TAPESTRY_HOST_SCHEDULER_H

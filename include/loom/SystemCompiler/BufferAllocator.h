#ifndef LOOM_SYSTEMCOMPILER_BUFFERALLOCATOR_H
#define LOOM_SYSTEMCOMPILER_BUFFERALLOCATOR_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Buffer Allocation
//===----------------------------------------------------------------------===//

/// Allocation descriptor for a single contract edge buffer.
struct BufferAllocation {
  std::string contractEdgeName;

  enum Location { SPM_PRODUCER, SPM_CONSUMER, SHARED_L2, EXTERNAL_DRAM };
  Location location = SPM_CONSUMER;

  /// Offset within the allocated memory region in bytes.
  uint64_t offsetBytes = 0;

  /// Total buffer size in bytes.
  uint64_t sizeBytes = 0;

  /// Number of data elements in the buffer.
  unsigned elementCount = 0;

  /// Whether double-buffering is active (two alternating buffers).
  bool doubleBuffered = false;

  /// For SHARED_L2: which bank the allocation starts in.
  std::optional<unsigned> l2BankIdx;

  /// For EXTERNAL_DRAM: virtual base address.
  std::optional<uint64_t> dramBaseAddr;

  /// Core instance index where this buffer resides.
  unsigned coreInstanceIdx = 0;
};

//===----------------------------------------------------------------------===//
// Per-core SPM Usage
//===----------------------------------------------------------------------===//

/// Tracks scratchpad memory usage for a single core.
struct CoreSPMUsage {
  std::string coreName;
  unsigned coreInstanceIdx = 0;
  uint64_t usedBytes = 0;
  uint64_t totalBytes = 0;
  double utilization = 0.0;
};

//===----------------------------------------------------------------------===//
// Buffer Allocation Plan
//===----------------------------------------------------------------------===//

/// Complete buffer allocation plan for all contract edges.
struct BufferAllocationPlan {
  std::vector<BufferAllocation> allocations;
  std::vector<CoreSPMUsage> coreSPMUsage;

  uint64_t l2UsedBytes = 0;
  uint64_t l2TotalBytes = 0;

  /// True if all buffers could be placed (DRAM fallback ensures feasibility).
  bool feasible = false;
};

//===----------------------------------------------------------------------===//
// Buffer Allocator
//===----------------------------------------------------------------------===//

/// Options for the buffer allocator.
struct BufferAllocatorOptions {
  /// Fraction of SPM to reserve for core-local use (not available for buffers).
  double spmReserveFraction = 0.2;

  /// Prefer double-buffering when the contract allows it.
  bool preferDoubleBuffering = true;

  /// L2 bank interleaving granularity in bytes.
  unsigned l2BankInterleaveBytes = 64;

  bool verbose = false;
};

/// Allocates buffers in SPM, L2, or DRAM for inter-core contract data.
///
/// Policy: SPM-first (consumer side), then shared L2, then DRAM fallback.
/// Contracts are sorted by data volume (largest first) for better packing.
/// When double-buffering is enabled and permitted by the contract, the
/// buffer size is doubled to allow overlapping transfer and compute.
class BufferAllocator {
public:
  /// Allocate buffers for all contract edges that require cross-core transfer.
  BufferAllocationPlan allocate(const AssignmentResult &assignment,
                                const std::vector<ContractSpec> &contracts,
                                const NoCSchedule &nocSchedule,
                                const SystemArchitecture &arch,
                                const BufferAllocatorOptions &opts);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_BUFFERALLOCATOR_H

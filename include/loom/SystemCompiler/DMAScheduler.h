#ifndef LOOM_SYSTEMCOMPILER_DMASCHEDULER_H
#define LOOM_SYSTEMCOMPILER_DMASCHEDULER_H

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/NoCScheduler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// DMA Transfer
//===----------------------------------------------------------------------===//

/// Descriptor for a single DMA transfer operation.
struct DMATransfer {
  std::string contractEdgeName;
  std::string srcCore;
  std::string dstCore;
  unsigned srcCoreIdx = 0;
  unsigned dstCoreIdx = 0;

  /// Source buffer allocation.
  BufferAllocation srcBuffer;

  /// Destination buffer allocation.
  BufferAllocation dstBuffer;

  /// Transfer size in bytes.
  uint64_t transferSizeBytes = 0;

  /// Timing relative to tile start.
  unsigned startCycle = 0;
  unsigned durationCycles = 0;
  unsigned endCycle = 0;

  /// Double-buffering slot (0 or 1).
  unsigned bufferSlot = 0;

  /// Whether this DMA overlaps with kernel execution.
  bool overlapWithCompute = false;
};

//===----------------------------------------------------------------------===//
// DMA Schedule
//===----------------------------------------------------------------------===//

/// Complete DMA schedule for all cross-core transfers within a tile.
struct DMASchedule {
  std::vector<DMATransfer> transfers;

  /// Kernel execution time in cycles.
  unsigned tileComputeCycles = 0;

  /// Total DMA time that does NOT overlap with compute.
  unsigned tileTransferCycles = 0;

  /// Effective tile time (compute + non-overlapped transfer).
  unsigned tileTotalCycles = 0;

  /// Fraction of total DMA time that overlaps with compute.
  double computeOverlapRatio = 0.0;
};

//===----------------------------------------------------------------------===//
// DMA Scheduler
//===----------------------------------------------------------------------===//

/// Options for the DMA scheduler.
struct DMASchedulerOptions {
  /// Estimated compute cycles per tile (used for overlap calculation).
  unsigned estimatedComputeCycles = 1000;

  bool verbose = false;
};

/// Plans DMA transfer operations for cross-core contract data movement.
///
/// For each cross-core transfer, the scheduler:
/// 1. Determines source and destination buffers from the allocation plan.
/// 2. Estimates transfer duration from data volume and NoC bandwidth.
/// 3. Identifies double-buffering overlap opportunities with computation.
/// 4. Computes effective tile timing including non-overlapped DMA cost.
class DMAScheduler {
public:
  /// Create a DMA schedule for all cross-core transfers.
  DMASchedule schedule(const BufferAllocationPlan &bufferPlan,
                       const NoCSchedule &nocSchedule,
                       const std::vector<ContractSpec> &contracts,
                       const AssignmentResult &assignment,
                       const SystemArchitecture &arch,
                       const DMASchedulerOptions &opts);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_DMASCHEDULER_H

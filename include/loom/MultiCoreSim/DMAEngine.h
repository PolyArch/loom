#ifndef LOOM_MULTICORESIM_DMAENGINE_H
#define LOOM_MULTICORESIM_DMAENGINE_H

#include "loom/MultiCoreSim/L2BankModel.h"
#include "loom/MultiCoreSim/SPMModel.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace loom {
namespace mcsim {

// Forward declarations.
class DRAMModel;

//===----------------------------------------------------------------------===//
// DMA Operation Descriptor
//===----------------------------------------------------------------------===//

/// Describes a DMA transfer operation to be submitted to the DMA engine.
struct DMAOpDescriptor {
  enum Type : uint8_t {
    SPM_TO_L2 = 0,
    L2_TO_SPM = 1,
    SPM_TO_DRAM = 2,
    DRAM_TO_SPM = 3,
    SPM_TO_SPM_VIA_L2 = 4,
  };

  Type type = SPM_TO_L2;
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;
  uint64_t srcOffset = 0;
  uint64_t dstOffset = 0;
  uint64_t sizeBytes = 0;
  std::function<void()> completionCallback;
};

//===----------------------------------------------------------------------===//
// Pending DMA Operation
//===----------------------------------------------------------------------===//

/// Internal state for a DMA operation in flight.
struct PendingDMAOp {
  enum State : uint8_t {
    SETUP = 0,
    TRANSFERRING = 1,
    COMPLETE = 2,
  };

  uint64_t id = 0;
  DMAOpDescriptor op;
  uint64_t startCycle = 0;
  uint64_t completionCycle = 0;
  State state = SETUP;
  uint64_t bytesTransferred = 0;
};

//===----------------------------------------------------------------------===//
// DMA Engine
//===----------------------------------------------------------------------===//

/// Per-core DMA engine that manages bulk data transfers.
///
/// Supports transfers between SPM<->L2 and SPM<->DRAM. Operates as a
/// state machine: SETUP -> TRANSFERRING -> COMPLETE. Supports multiple
/// concurrent in-flight operations for double-buffering.
class DMAEngine {
public:
  DMAEngine(unsigned coreId, unsigned maxInflight, unsigned setupLatency,
            unsigned transferBandwidthBytesPerCycle);

  /// Submit a new DMA operation. Returns the operation ID.
  /// Returns 0 if the engine has no capacity (all slots busy).
  uint64_t submit(const DMAOpDescriptor &op, uint64_t globalCycle);

  /// Advance the DMA engine state by one cycle.
  /// Progresses all in-flight operations through their state machines.
  void tick(uint64_t globalCycle, std::vector<SPMModel> &spms,
            std::vector<L2BankModel> &l2Banks, DRAMModel &dram,
            unsigned l2NumBanks, uint64_t l2BankSizeBytes);

  /// Check if a specific DMA operation has completed.
  bool isComplete(uint64_t dmaId) const;

  /// Check if the engine can accept another DMA operation.
  bool hasCapacity() const;

  /// Number of currently in-flight operations.
  unsigned inflightCount() const;

  unsigned getCoreId() const { return coreId_; }
  uint64_t getTotalBytesTransferred() const { return totalBytesTransferred_; }
  uint64_t getTotalTransferCycles() const { return totalTransferCycles_; }
  uint64_t getOpsSubmitted() const { return opsSubmitted_; }
  uint64_t getOpsCompleted() const { return opsCompleted_; }

private:
  /// Compute the transfer duration in cycles for a given size.
  uint64_t computeTransferCycles(uint64_t sizeBytes) const;

  /// Execute the actual data transfer for a completed operation.
  void executeTransfer(PendingDMAOp &pendingOp, std::vector<SPMModel> &spms,
                       std::vector<L2BankModel> &l2Banks, DRAMModel &dram,
                       unsigned l2NumBanks, uint64_t l2BankSizeBytes);

  unsigned coreId_;
  unsigned maxInflight_;
  unsigned setupLatency_;
  unsigned transferBandwidthBytesPerCycle_;

  uint64_t nextDMAId_ = 1;

  /// In-flight DMA operations.
  std::deque<PendingDMAOp> inflightOps_;

  /// Recently completed operation IDs (retained for isComplete queries).
  std::deque<uint64_t> completedIds_;

  /// Statistics.
  uint64_t totalBytesTransferred_ = 0;
  uint64_t totalTransferCycles_ = 0;
  uint64_t opsSubmitted_ = 0;
  uint64_t opsCompleted_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_DMAENGINE_H

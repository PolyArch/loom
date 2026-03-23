#ifndef LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H
#define LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H

#include "loom/MultiCoreSim/DMAEngine.h"
#include "loom/MultiCoreSim/DRAMModel.h"
#include "loom/MultiCoreSim/L2BankModel.h"
#include "loom/MultiCoreSim/SPMModel.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Memory Hierarchy Configuration
//===----------------------------------------------------------------------===//

/// Configuration parameters for the entire memory hierarchy.
struct MemoryHierarchyConfig {
  /// Number of cores in the system.
  unsigned numCores = 1;

  /// Per-core SPM size in bytes.
  uint64_t spmSizeBytes = 4096;

  /// SPM access latency in cycles (typically 1 for local core).
  unsigned spmAccessLatency = 1;

  /// Total shared L2 size in bytes.
  uint64_t l2SizeBytes = 262144; // 256KB

  /// Number of L2 banks.
  unsigned l2NumBanks = 4;

  /// L2 bank access latency in cycles (excluding NoC).
  unsigned l2BankAccessLatency = 4;

  /// L2 bank width in bytes (interleave granularity).
  unsigned l2BankWidthBytes = 32;

  /// External DRAM size in bytes.
  uint64_t dramSizeBytes = 1ULL << 30; // 1GB

  /// DRAM access latency in cycles.
  unsigned dramAccessLatency = 100;

  /// DRAM bandwidth in bytes per cycle.
  unsigned dramBandwidthBytesPerCycle = 16;

  /// Maximum concurrent DMA operations per core.
  unsigned dmaMaxInflight = 4;

  /// DMA setup latency in cycles.
  unsigned dmaSetupLatency = 5;

  /// DMA transfer bandwidth in bytes per cycle.
  unsigned dmaTransferBandwidthBytesPerCycle = 16;
};

//===----------------------------------------------------------------------===//
// Memory Access Result
//===----------------------------------------------------------------------===//

/// Result of a memory access operation.
struct MemoryAccessResult {
  bool hit = false;
  unsigned latency = 0;
  uint64_t requestId = 0;
};

//===----------------------------------------------------------------------===//
// Memory Hierarchy Statistics
//===----------------------------------------------------------------------===//

/// Aggregate statistics for the memory hierarchy.
struct MemStats {
  uint64_t spmReads = 0;
  uint64_t spmWrites = 0;
  uint64_t l2Reads = 0;
  uint64_t l2Writes = 0;
  uint64_t l2BankConflicts = 0;
  uint64_t dramReads = 0;
  uint64_t dramWrites = 0;
  uint64_t dmaOpsSubmitted = 0;
  uint64_t dmaOpsCompleted = 0;
  uint64_t dmaTotalBytesTransferred = 0;
  uint64_t dmaTotalCycles = 0;
  double dmaOverlapRatio = 0.0;
};

//===----------------------------------------------------------------------===//
// Memory Hierarchy Model
//===----------------------------------------------------------------------===//

/// Top-level memory hierarchy model integrating SPM, L2, DMA, and DRAM.
///
/// Routes memory accesses to the correct subsystem and orchestrates
/// per-cycle advancement of all memory components.
class MemoryHierarchyModel {
public:
  explicit MemoryHierarchyModel(const MemoryHierarchyConfig &config);
  ~MemoryHierarchyModel();

  MemoryHierarchyModel(const MemoryHierarchyModel &) = delete;
  MemoryHierarchyModel &operator=(const MemoryHierarchyModel &) = delete;

  // --- Initialization ---

  /// Initialize per-core SPM contents.
  void initializeSPM(unsigned coreId, const std::vector<uint8_t> &data);

  /// Initialize shared L2 contents (distributed across banks).
  void initializeL2(const std::vector<uint8_t> &data);

  /// Initialize DRAM contents at a given base address.
  void initializeDRAM(uint64_t baseAddr, const std::vector<uint8_t> &data);

  // --- SPM access (single-cycle, local core only) ---

  uint64_t spmRead(unsigned coreId, uint64_t offset, unsigned sizeBytes);
  void spmWrite(unsigned coreId, uint64_t offset, uint64_t data,
                unsigned sizeBytes);

  // --- L2 access (multi-cycle, banked) ---

  MemoryAccessResult l2Read(unsigned requestingCoreId, uint64_t offset,
                            unsigned sizeBytes);
  MemoryAccessResult l2Write(unsigned requestingCoreId, uint64_t offset,
                             uint64_t data, unsigned sizeBytes);

  // --- DMA operations ---

  /// Submit a DMA transfer. Returns the DMA operation ID (0 if no capacity).
  uint64_t submitDMA(unsigned coreId, const DMAOpDescriptor &op);

  /// Check if a DMA operation has completed.
  bool isDMAComplete(unsigned coreId, uint64_t dmaId) const;

  // --- Per-cycle update ---

  /// Advance all memory hierarchy components by one cycle.
  void tick(uint64_t globalCycle);

  // --- Statistics ---

  MemStats getStats() const;

  // --- Accessors for sub-models ---

  SPMModel &getSPM(unsigned coreId) { return spms_[coreId]; }
  const SPMModel &getSPM(unsigned coreId) const { return spms_[coreId]; }

  L2BankModel &getL2Bank(unsigned bankId) { return l2Banks_[bankId]; }
  const L2BankModel &getL2Bank(unsigned bankId) const {
    return l2Banks_[bankId];
  }

  DRAMModel &getDRAM() { return dram_; }
  const DRAMModel &getDRAM() const { return dram_; }

  const MemoryHierarchyConfig &getConfig() const { return config_; }

private:
  /// Decode an L2 address into bank index and bank-local offset.
  std::pair<unsigned, uint64_t> decodeL2Address(uint64_t offset) const;

  MemoryHierarchyConfig config_;

  /// Per-core scratchpad memories.
  std::vector<SPMModel> spms_;

  /// Shared L2 banks.
  std::vector<L2BankModel> l2Banks_;

  /// External DRAM.
  DRAMModel dram_;

  /// Per-core DMA engines.
  std::vector<DMAEngine> dmaEngines_;

  /// Current global cycle (updated in tick()).
  uint64_t currentCycle_ = 0;

  /// Accumulated statistics.
  uint64_t spmReadCount_ = 0;
  uint64_t spmWriteCount_ = 0;
  uint64_t l2ReadCount_ = 0;
  uint64_t l2WriteCount_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H

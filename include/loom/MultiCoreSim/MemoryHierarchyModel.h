#ifndef LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H
#define LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H

#include "loom/MultiCoreSim/TapestryTypes.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace mcsim {

// Configuration for the memory hierarchy model.
struct MemoryHierarchyConfig {
  uint64_t spmSizeBytes = 64 * 1024;   // 64 KB per-core SPM
  unsigned spmLatencyCycles = 1;
  uint64_t l2SizeBytes = 512 * 1024;   // 512 KB shared L2
  unsigned l2LatencyCycles = 10;
  unsigned dramLatencyCycles = 100;
  unsigned numCores = 1;
};

// Represents a pending memory request in the hierarchy.
struct MemoryHierarchyRequest {
  uint64_t requestId = 0;
  unsigned coreId = 0;
  uint64_t address = 0;
  uint64_t data = 0;
  unsigned byteWidth = 0;
  bool isStore = false;
  uint64_t issueCycle = 0;
  uint64_t completionCycle = 0;
};

// Models a two-level memory hierarchy: per-core SPM + shared L2 + DRAM.
// The SPM is modeled as a directly-mapped region (address-based),
// and the L2 is a simple capacity-based model.
class MemoryHierarchyModel {
public:
  explicit MemoryHierarchyModel(const MemoryHierarchyConfig &config);
  ~MemoryHierarchyModel();

  MemoryHierarchyModel(const MemoryHierarchyModel &) = delete;
  MemoryHierarchyModel &operator=(const MemoryHierarchyModel &) = delete;

  // Issue a memory request. Returns the estimated completion cycle.
  uint64_t issueRequest(unsigned coreId, uint64_t address, unsigned byteWidth,
                        bool isStore, uint64_t currentCycle);

  // Advance the model by one cycle.
  void stepOneCycle();

  // Get the current cycle.
  uint64_t getCurrentCycle() const { return currentCycle_; }

  // Gather statistics.
  MemoryHierarchyStats getStats() const;

  // Check if an address falls within the SPM range for a given core.
  bool isInSpmRange(unsigned coreId, uint64_t address) const;

private:
  MemoryHierarchyConfig config_;
  uint64_t currentCycle_ = 0;
  uint64_t nextRequestId_ = 0;

  // Per-core SPM base addresses (each core has a distinct region).
  std::vector<uint64_t> spmBaseAddresses_;

  // Statistics counters.
  uint64_t spmHits_ = 0;
  uint64_t spmMisses_ = 0;
  uint64_t l2Hits_ = 0;
  uint64_t l2Misses_ = 0;
  uint64_t dramAccesses_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_MEMORYHIERARCHYMODEL_H

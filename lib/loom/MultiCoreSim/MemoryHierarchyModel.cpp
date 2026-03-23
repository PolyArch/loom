#include "loom/MultiCoreSim/MemoryHierarchyModel.h"

#include <algorithm>

namespace loom {
namespace mcsim {

MemoryHierarchyModel::MemoryHierarchyModel(const MemoryHierarchyConfig &config)
    : config_(config) {
  // Each core gets a non-overlapping SPM address range.
  spmBaseAddresses_.resize(config_.numCores);
  for (unsigned core = 0; core < config_.numCores; ++core) {
    spmBaseAddresses_[core] = static_cast<uint64_t>(core) * config_.spmSizeBytes;
  }
}

MemoryHierarchyModel::~MemoryHierarchyModel() = default;

uint64_t MemoryHierarchyModel::issueRequest(unsigned coreId, uint64_t address,
                                            unsigned byteWidth, bool isStore,
                                            uint64_t currentCycle) {
  (void)byteWidth;
  (void)isStore;

  ++nextRequestId_;

  // Check if the address falls in the core's SPM range.
  if (isInSpmRange(coreId, address)) {
    ++spmHits_;
    return currentCycle + config_.spmLatencyCycles;
  }

  // Not in SPM; check L2. Use a simple modular heuristic: even
  // addresses hit, odd addresses miss (for basic modeling).
  ++spmMisses_;

  // Simple L2 hit/miss model: addresses within L2 range are hits.
  if (address < config_.l2SizeBytes) {
    ++l2Hits_;
    return currentCycle + config_.l2LatencyCycles;
  }

  // L2 miss: go to DRAM.
  ++l2Misses_;
  ++dramAccesses_;
  return currentCycle + config_.dramLatencyCycles;
}

void MemoryHierarchyModel::stepOneCycle() { ++currentCycle_; }

MemoryHierarchyStats MemoryHierarchyModel::getStats() const {
  MemoryHierarchyStats stats;
  stats.spmHits = spmHits_;
  stats.spmMisses = spmMisses_;
  stats.l2Hits = l2Hits_;
  stats.l2Misses = l2Misses_;
  stats.dramAccesses = dramAccesses_;
  return stats;
}

bool MemoryHierarchyModel::isInSpmRange(unsigned coreId,
                                        uint64_t address) const {
  if (coreId >= spmBaseAddresses_.size())
    return false;
  uint64_t base = spmBaseAddresses_[coreId];
  return address >= base && address < base + config_.spmSizeBytes;
}

} // namespace mcsim
} // namespace loom

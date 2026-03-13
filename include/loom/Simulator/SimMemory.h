//===-- SimMemory.h - Simulated fabric.memory/extmemory ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Memory model per spec-fabric-mem.md. Supports both on-chip (fabric.memory)
// and external (fabric.extmemory) memory with tag-based addressing,
// region-based offset tables, and configurable latency.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMMEMORY_H
#define LOOM_SIMULATOR_SIMMEMORY_H

#include "loom/Simulator/SimModule.h"

#include <deque>
#include <unordered_map>

namespace loom {
namespace sim {

class SimMemory : public SimModule {
public:
  SimMemory(bool isExternal, unsigned ldCount, unsigned stCount,
            unsigned dataWidth, unsigned tagWidth, unsigned addrWidth,
            unsigned numRegion = 1, uint32_t extLatency = 0);

  bool isCombinational() const override { return false; }
  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  /// Set the backing memory for external memory simulation.
  void setBackingMemory(uint8_t *mem, size_t size) {
    backingMem_ = mem;
    backingMemSize_ = size;
  }

private:
  bool isExternal_;
  unsigned ldCount_;
  unsigned stCount_;
  unsigned dataWidth_;
  unsigned tagWidth_;
  unsigned addrWidth_;
  unsigned numRegion_;
  uint32_t extLatency_;

  /// On-chip memory storage (for fabric.memory).
  std::vector<uint8_t> onChipMem_;
  static constexpr size_t kDefaultOnChipSize = 4096;

  /// External memory backing store.
  uint8_t *backingMem_ = nullptr;
  size_t backingMemSize_ = 0;

  /// Pending load responses (for latency modeling).
  struct PendingLoad {
    uint64_t data = 0;
    uint16_t tag = 0;
    uint64_t readyCycle = 0;
    unsigned laneIdx = 0;
  };
  std::deque<PendingLoad> pendingLoads_;

  /// addr_offset_table: per-region tag-to-offset mapping.
  /// Per spec-fabric-mem.md: each region has valid, start_tag, end_tag,
  /// addr_offset. A tag matches region i if valid && start_tag <= tag < end_tag.
  struct AddrOffsetEntry {
    bool valid = false;
    uint16_t startTag = 0;
    uint16_t endTag = 0;
    uint64_t addrOffset = 0;
  };
  std::vector<AddrOffsetEntry> addrOffsetTable_;

  /// Port layout per spec-fabric-mem.md (multi-port memory):
  /// For ldCount=L, stCount=S:
  ///   Inputs:  [ld_addr_0..L-1, st_addr_0..S-1, st_data_0..S-1]
  ///   Outputs: [ld_data_0..L-1, ld_done_0..L-1, st_done_0..S-1]

  /// Read from memory (on-chip or external).
  uint64_t memRead(uint64_t addr) const;

  /// Write to memory (on-chip or external).
  void memWrite(uint64_t addr, uint64_t data);

  /// Resolve address through addr_offset_table. Returns resolved address
  /// and sets matched=true if a region matched, false otherwise.
  uint64_t resolveAddr(uint64_t addr, uint16_t tag, bool &matched) const;

  /// Store pairing queue: per-tag (or global for stCount==1) FIFO pairing
  /// of staddr and stdata arrivals.
  struct StorePairEntry {
    uint64_t addr = 0;
    uint64_t data = 0;
    bool hasAddr = false;
    bool hasData = false;
    unsigned laneIdx = 0;
  };
  std::unordered_map<uint16_t, std::deque<StorePairEntry>> storePairQueues_;

  /// Deadlock detection counters per store tag.
  std::unordered_map<uint16_t, uint32_t> storeDeadlockCounters_;
  static constexpr uint32_t kDeadlockTimeout = 65535;

  /// Pending store-done tokens (from paired store commits in advanceClock).
  struct PendingStDone {
    uint16_t tag = 0;
    unsigned laneIdx = 0;
  };
  std::vector<std::deque<PendingStDone>> pendingStDone_;

  bool firedThisCycle_ = false;
  uint64_t currentSimCycle_ = 0;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMMEMORY_H

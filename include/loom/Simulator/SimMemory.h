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
            uint32_t extLatency = 0);

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

  /// Port layout (multi-port memory):
  /// For ldCount=L, stCount=S:
  ///   Inputs:  [ld_addr_0, ..., ld_addr_{L-1}, st_data_0, st_addr_0, ..., st_data_{S-1}, st_addr_{S-1}]
  ///   Outputs: [ld_data_0, ..., ld_data_{L-1}, ld_done_0, ..., ld_done_{L-1}, st_done_0, ..., st_done_{S-1}]
  ///
  /// For single-port (ldCount<=1, stCount<=1), layout is simpler.

  /// Read from memory (on-chip or external).
  uint64_t memRead(uint64_t addr) const;

  /// Write to memory (on-chip or external).
  void memWrite(uint64_t addr, uint64_t data);

  bool firedThisCycle_ = false;
  uint64_t currentSimCycle_ = 0;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMMEMORY_H

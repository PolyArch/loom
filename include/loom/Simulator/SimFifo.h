//===-- SimFifo.h - Simulated fabric.fifo -----------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// FIFO model per spec-fabric-fifo.md. Supports bypassable and non-bypassable
// modes. Bypassable FIFOs act as combinational wires when bypass is enabled.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMFIFO_H
#define LOOM_SIMULATOR_SIMFIFO_H

#include "loom/Simulator/SimModule.h"

#include <deque>

namespace loom {
namespace sim {

class SimFifo : public SimModule {
public:
  SimFifo(unsigned depth, bool bypassable);

  /// Bypassable FIFOs in bypass mode are combinational.
  bool isCombinational() const override { return bypassed_; }

  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  struct FifoEntry {
    uint64_t data = 0;
    uint16_t tag = 0;
    bool hasTag = false;
  };

  unsigned depth_;
  bool bypassable_;
  bool bypassed_ = false;
  std::deque<FifoEntry> buffer_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMFIFO_H

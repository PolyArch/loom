//===-- SimTemporalSW.h - Simulated fabric.temporal_sw -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Temporal switch model per spec-fabric-temporal_sw.md.
// Tag-aware routing with per-output round-robin arbitration and
// atomic broadcast semantics.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMTEMPORALSW_H
#define LOOM_SIMULATOR_SIMTEMPORALSW_H

#include "loom/Simulator/SimModule.h"

namespace loom {
namespace sim {

class SimTemporalSW : public SimModule {
public:
  SimTemporalSW(unsigned numInputs, unsigned numOutputs, unsigned tagWidth,
                unsigned numRouteTable,
                const std::vector<bool> &connectivityTable);

  /// Sequential: has round-robin arbitration state.
  bool isCombinational() const override { return false; }

  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  unsigned numInputs_;
  unsigned numOutputs_;
  unsigned tagWidth_;
  unsigned numRouteTable_;
  std::vector<bool> connectivity_;

  /// Route table slots.
  struct RouteSlot {
    bool valid = false;
    uint16_t tag = 0;
    /// Per connected position: true = route enabled.
    std::vector<bool> routes;
  };
  std::vector<RouteSlot> routeSlots_;

  /// Per-output round-robin pointer (shared across tag slots).
  std::vector<unsigned> rrPointer_;

  /// Derived: for each slot, for each output, which input is routed.
  /// -1 if no input routes to this output in this slot.
  struct SlotRouting {
    std::vector<int> outputSource;        // [numOutputs] -> input idx
    std::vector<std::vector<unsigned>> inputTargets; // [numInputs] -> output idxs
  };
  std::vector<SlotRouting> slotRoutings_;

  /// Per-input matched slot index (-1 if no match).
  std::vector<int> inputSlotMatch_;

  /// Per-output arbitration winner for this cycle (-1 if no winner).
  std::vector<int> arbWinner_;

  /// Whether each input won broadcast for all its targets.
  std::vector<bool> broadcastOk_;

  void rebuildSlotRouting();
  void performArbitration();
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTEMPORALSW_H

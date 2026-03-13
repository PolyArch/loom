//===-- SimSwitch.h - Simulated fabric.switch --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Combinational static switch model per spec-fabric-switch.md.
// Routes input channels to output channels based on configured route_table.
// Supports broadcast (one input to multiple outputs) with atomic handshake.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMSWITCH_H
#define LOOM_SIMULATOR_SIMSWITCH_H

#include "loom/Simulator/SimModule.h"

namespace loom {
namespace sim {

class SimSwitch : public SimModule {
public:
  SimSwitch(unsigned numInputs, unsigned numOutputs,
            const std::vector<bool> &connectivityTable);

  bool isCombinational() const override { return true; }
  void evaluateCombinational() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  unsigned numInputs_;
  unsigned numOutputs_;

  /// Connectivity table: [numOutputs][numInputs], row-major.
  /// True if physical connection exists.
  std::vector<bool> connectivity_;

  /// Route table: indexed by connected position in connectivity table.
  /// True = route enabled, False = route disabled.
  std::vector<bool> routeTable_;

  /// Derived: for each output, which input is routed to it (-1 if none).
  std::vector<int> outputSource_;

  /// Derived: for each input, which outputs it broadcasts to.
  std::vector<std::vector<unsigned>> inputTargets_;

  /// Rebuild derived routing maps from routeTable_ and connectivity_.
  void rebuildRouting();
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMSWITCH_H

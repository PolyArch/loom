//===-- SimModule.h - Base class for simulated fabric modules ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Abstract base class for all fabric module simulation models. Each simulated
// module implements two-phase evaluation: combinational logic (valid/ready/data
// propagation) and sequential state advance (clock edge behavior).
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMMODULE_H
#define LOOM_SIMULATOR_SIMMODULE_H

#include "loom/Simulator/SimTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace sim {

/// Base class for all simulated fabric modules.
///
/// Modules read from input channels and write to output channels.
/// Forward signals (valid, data, tag) are set by the producing module.
/// The ready signal is set by the consuming module.
///
/// Evaluation order within a cycle:
///   1. evaluateCombinational() - compute output valid/data from input valid/data
///                                and compute input ready from output ready.
///   2. advanceClock() - consume transferred tokens, update internal state.
class SimModule {
public:
  virtual ~SimModule() = default;

  /// Whether this module is purely combinational (evaluated only in phase 1).
  /// Sequential modules are evaluated in both phases.
  virtual bool isCombinational() const = 0;

  /// Evaluate combinational logic: read input valid/data/tag and output ready,
  /// then compute output valid/data/tag and input ready.
  /// Called repeatedly until fixed-point convergence in phase 1.
  virtual void evaluateCombinational() = 0;

  /// Advance sequential state on the clock edge. Called once per cycle in
  /// phase 2. Consumes transferred tokens (where valid && ready) and updates
  /// internal state (FIFOs, operand buffers, arbitration pointers, etc.).
  virtual void advanceClock() {}

  /// Reset all internal state. Called during configuration.
  virtual void reset() = 0;

  /// Apply configuration from raw config words (extracted from config.bin).
  /// The words correspond to this module's config_mem slice.
  virtual void configure(const std::vector<uint32_t> &configWords) = 0;

  /// Collect trace events for the current cycle.
  virtual void collectTraceEvents(std::vector<TraceEvent> &events,
                                  uint64_t cycle) = 0;

  /// Get current performance counters.
  virtual PerfSnapshot getPerfSnapshot() const = 0;

  //===--------------------------------------------------------------------===//
  // Port access
  //===--------------------------------------------------------------------===//

  /// Input channels (driven by upstream producers).
  std::vector<SimChannel *> inputs;

  /// Output channels (driven by this module).
  std::vector<SimChannel *> outputs;

  //===--------------------------------------------------------------------===//
  // Identity
  //===--------------------------------------------------------------------===//

  /// Hardware node ID in the ADG graph.
  uint32_t hwNodeId = 0;

  /// Human-readable name (e.g., "sw_0", "tpe_1").
  std::string name;

  /// Operation name from the MLIR op (e.g., "fabric.switch").
  std::string opName;

protected:
  //===--------------------------------------------------------------------===//
  // Sticky error state (per spec-fabric-error.md)
  //===--------------------------------------------------------------------===//

  /// Latch an error code. Only the first error is captured (sticky).
  /// When multiple errors occur in the same cycle, the numerically
  /// smallest code takes precedence.
  void latchError(uint16_t code) {
    if (code == RtError::OK)
      return;
    if (!errorValid_ || code < errorCode_) {
      errorValid_ = true;
      errorCode_ = code;
    }
  }

  bool errorValid_ = false;
  uint16_t errorCode_ = RtError::OK;

  //===--------------------------------------------------------------------===//
  // Performance counters
  //===--------------------------------------------------------------------===//

  PerfSnapshot perf_;
};

/// Factory function: create a SimModule from an ADG node's attributes.
/// Returns nullptr if the op name is not recognized.
std::unique_ptr<SimModule> createSimModule(
    uint32_t hwNodeId, const std::string &name, const std::string &opName,
    unsigned numInputs, unsigned numOutputs,
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::vector<std::pair<std::string, std::string>> &strAttrs);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMMODULE_H

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

  /// Audit route configuration. Returns diagnostics for unrouted or
  /// misconfigured ports. connectedInputs[i] is true if input port i was
  /// wired by an ADG edge (not a dummy channel).
  virtual void auditRoutes(const std::vector<bool> &connectedInputs,
                           std::vector<AuditDiagnostic> &diags) const {
    (void)connectedInputs;
    (void)diags;
  }

  /// Returns true if the given input port index has a configured route.
  /// Default: always true (non-switch modules don't have routing).
  virtual bool inputHasRoute(unsigned portIdx) const {
    (void)portIdx;
    return true;
  }

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

  //===--------------------------------------------------------------------===//
  // Sticky error state (per spec-fabric-error.md)
  //===--------------------------------------------------------------------===//

  /// Report an error detected during combinational evaluation.
  /// Multiple calls per cycle select the numerically smallest code.
  /// Call commitError() once per cycle to finalize sticky state.
  void latchError(uint16_t code) {
    if (code == RtError::OK)
      return;
    if (errorValid_)
      return; // Already sticky from a prior cycle.
    if (!pendingError_ || code < pendingErrorCode_) {
      pendingError_ = true;
      pendingErrorCode_ = code;
    }
  }

  /// Returns true if a sticky error is latched.
  bool hasError() const { return errorValid_; }

  /// Returns the sticky error code (RtError::OK if none).
  uint16_t getErrorCode() const { return errorCode_; }

  /// Finalize pending error into sticky state. Called once per cycle
  /// after all combinational evaluation is complete.
  void commitError() {
    if (pendingError_ && !errorValid_) {
      errorValid_ = true;
      errorCode_ = pendingErrorCode_;
    }
    pendingError_ = false;
    pendingErrorCode_ = RtError::OK;
  }

protected:
  bool errorValid_ = false;
  uint16_t errorCode_ = RtError::OK;
  bool pendingError_ = false;
  uint16_t pendingErrorCode_ = RtError::OK;

  //===--------------------------------------------------------------------===//
  // Performance counters
  //===--------------------------------------------------------------------===//

  PerfSnapshot perf_;
};

/// Factory function: create a SimModule from an ADG node's attributes.
/// Returns nullptr if the op name is not recognized.
/// \p defaultExtLatency is used for fabric.extmemory when the ADG node
/// does not provide an explicit ext_latency attribute.
std::unique_ptr<SimModule> createSimModule(
    uint32_t hwNodeId, const std::string &name, const std::string &opName,
    unsigned numInputs, unsigned numOutputs,
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::vector<std::pair<std::string, std::string>> &strAttrs,
    const std::vector<std::pair<std::string, std::vector<int8_t>>>
        &arrayAttrs = {},
    uint32_t defaultExtLatency = 10);

/// Compute the CONFIG_WIDTH (in bits) for a fabric module based on its
/// attributes. Implements the formula table from spec-fabric-config_mem.md.
unsigned computeConfigWidth(
    const std::string &opName,
    unsigned numInputs, unsigned numOutputs,
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::vector<std::pair<std::string, std::string>> &strAttrs,
    const std::vector<std::pair<std::string, std::vector<int8_t>>>
        &arrayAttrs = {});

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMMODULE_H

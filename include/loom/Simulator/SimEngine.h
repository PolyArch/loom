//===-- SimEngine.h - Two-phase cycle-accurate simulation engine -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// The simulation engine orchestrates per-cycle two-phase evaluation of a
// mapped DFG on a configured ADG:
//   Phase 1: Combinational convergence (topological order, fixed-point)
//   Phase 2: Sequential state advance (clock edge)
//
// The engine reads a configured ADG (fabric.module MLIR), builds SimModule
// instances for each hardware node, wires them via SimChannels, loads
// config.bin, and runs the simulation loop.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMENGINE_H
#define LOOM_SIMULATOR_SIMENGINE_H

#include "loom/Mapper/Graph.h"
#include "loom/Simulator/SimModule.h"
#include "loom/Simulator/SimTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace sim {

/// Result of a simulation run.
struct SimResult {
  bool success = false;
  uint64_t totalCycles = 0;
  uint64_t configCycles = 0;
  std::string errorMessage;

  /// Per-node performance snapshots (indexed by hwNodeId).
  std::vector<PerfSnapshot> nodePerf;

  /// Collected trace events (empty if TraceMode::Off or Summary).
  std::vector<TraceEvent> traceEvents;
};

/// The simulation engine.
class SimEngine {
public:
  explicit SimEngine(const SimConfig &config = SimConfig());

  /// Build the simulation model from an ADG graph.
  /// The graph should be a flattened ADG (from ADGFlattener).
  bool buildFromGraph(const Graph &adg);

  /// Load configuration from a config.bin file.
  bool loadConfig(const std::string &configBinPath);

  /// Load configuration from raw bytes.
  bool loadConfig(const std::vector<uint8_t> &configBlob);

  /// Set input data for a specific input port of the accelerator.
  /// portIdx: index of the module input port (boundary input).
  void setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                const std::vector<uint16_t> &tags = {});

  /// Get output data from a specific output port.
  std::vector<uint64_t> getOutput(unsigned portIdx) const;

  /// Get output tags from a specific output port.
  std::vector<uint16_t> getOutputTags(unsigned portIdx) const;

  /// Run the simulation until completion or timeout.
  SimResult run();

  /// Reset the engine for a new invocation (preserves configuration).
  void resetExecution();

  /// Full reset (clears configuration too).
  void resetAll();

  /// Get the current cycle count.
  uint64_t getCurrentCycle() const { return currentCycle_; }

private:
  SimConfig config_;
  uint64_t currentCycle_ = 0;
  uint32_t epochId_ = 0;
  uint64_t invocationId_ = 0;

  /// All channels (owned storage).
  std::vector<std::unique_ptr<SimChannel>> channels_;

  /// All simulated modules.
  std::vector<std::unique_ptr<SimModule>> modules_;

  /// Topological order for combinational modules (phase 1).
  std::vector<SimModule *> combOrder_;

  /// All modules that have sequential state (phase 2).
  std::vector<SimModule *> seqModules_;

  /// Boundary input channels (from host to accelerator).
  std::vector<SimChannel *> boundaryInputs_;

  /// Boundary output channels (from accelerator to host).
  std::vector<SimChannel *> boundaryOutputs_;

  /// Input data queues for boundary inputs.
  struct InputQueue {
    std::vector<uint64_t> data;
    std::vector<uint16_t> tags;
    size_t pos = 0;
    bool hasTag = false;
  };
  std::vector<InputQueue> inputQueues_;

  /// Output data collectors for boundary outputs.
  struct OutputCollector {
    std::vector<uint64_t> data;
    std::vector<uint16_t> tags;
  };
  std::vector<OutputCollector> outputCollectors_;

  /// Trace event buffer for current cycle.
  std::vector<TraceEvent> cycleEvents_;

  /// Full trace buffer.
  std::vector<TraceEvent> allTraceEvents_;

  /// Config blob for config_mem loader.
  std::vector<uint8_t> configBlob_;

  /// Per-module config address map entry.
  struct ModuleConfigSlice {
    uint32_t wordOffset = 0;
    uint32_t wordCount = 0;
  };
  std::vector<ModuleConfigSlice> moduleConfigMap_;

  /// Whether a combinational loop was detected (Kahn's didn't visit all).
  bool hasCombLoop_ = false;

  /// Compute topological order of combinational modules.
  void computeTopologicalOrder();

  /// Run one simulation cycle (both phases).
  void stepOneCycle();

  /// Check if simulation is complete (all outputs drained).
  bool isComplete() const;

  /// Drive boundary input channels from queues (sets valid/data before eval).
  void driveBoundaryInputs();

  /// Set boundary output channels ready before combinational eval.
  void driveBoundaryOutputReady();

  /// After eval, advance input queues and collect outputs based on handshake.
  void advanceBoundaryState();

  /// Emit trace events for this cycle.
  void emitTraceEvents();
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMENGINE_H

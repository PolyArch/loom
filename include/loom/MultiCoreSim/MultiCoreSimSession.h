#ifndef LOOM_MULTICORESIM_MULTICORESIMSESSION_H
#define LOOM_MULTICORESIM_MULTICORESIMSESSION_H

#include "loom/MultiCoreSim/CoreSimWrapper.h"
#include "loom/MultiCoreSim/InterCoreEvent.h"
#include "loom/Simulator/SimTypes.h"
#include "loom/Simulator/StaticModel.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Multi-Core Simulation Configuration
//===----------------------------------------------------------------------===//

/// Configuration for a multi-core simulation run.
struct MultiCoreSimConfig {
  /// Maximum number of global cycles before timeout.
  uint64_t maxGlobalCycles = 1000000;

  /// Enable NoC contention modeling (bandwidth limits, serialization).
  bool enableNoCContention = true;

  /// Enable memory hierarchy simulation (SPM, L2, DRAM latencies).
  bool enableMemoryHierarchy = true;

  /// Enable cycle-level tracing.
  bool enableTracing = false;

  /// Cycle range for tracing (only relevant when enableTracing is true).
  uint64_t traceStartCycle = 0;
  uint64_t traceEndCycle = UINT64_MAX;

  /// Default NoC transfer latency per hop (in cycles).
  unsigned nocLatencyPerHop = 2;

  /// NoC flit width in bits.
  unsigned nocFlitWidthBits = 32;

  /// NoC link bandwidth in flits per cycle.
  unsigned nocLinkBandwidth = 1;

  /// DMA transfer latency overhead in cycles.
  unsigned dmaLatencyOverheadCycles = 10;

  /// External memory (DRAM) access latency in cycles.
  unsigned dramLatencyCycles = 100;

  /// L2 cache access latency in cycles.
  unsigned l2LatencyCycles = 10;
};

//===----------------------------------------------------------------------===//
// Multi-Core Simulation Result
//===----------------------------------------------------------------------===//

/// Per-core result within a multi-core simulation.
struct CoreSimResultEntry {
  std::string coreName;
  std::string coreType;
  unsigned coreIdx = 0;

  uint64_t activeCycles = 0;
  uint64_t stallCycles = 0;
  uint64_t idleCycles = 0;
  double utilization = 0.0;

  /// The underlying single-core simulation result.
  sim::SimResult perCoreResult;
};

/// NoC-level statistics from the simulation.
struct NoCSimStats {
  uint64_t totalFlitsTransferred = 0;
  uint64_t totalTransferCycles = 0;
  double avgLinkUtilization = 0.0;
  double maxLinkUtilization = 0.0;
  uint64_t contentionStallCycles = 0;
};

/// Memory hierarchy statistics from the simulation.
struct MemSimStats {
  uint64_t spmReads = 0;
  uint64_t spmWrites = 0;
  uint64_t l2Reads = 0;
  uint64_t l2Writes = 0;
  uint64_t dramReads = 0;
  uint64_t dramWrites = 0;
  uint64_t dmaTotalBytes = 0;
  uint64_t dmaTotalCycles = 0;
};

/// Complete result of a multi-core simulation run.
struct MultiCoreSimResult {
  bool success = false;
  std::string errorMessage;

  /// Total global cycles elapsed.
  uint64_t totalGlobalCycles = 0;

  /// Per-core detailed results.
  std::vector<CoreSimResultEntry> coreResults;

  /// NoC statistics.
  NoCSimStats nocStats;

  /// Memory hierarchy statistics.
  MemSimStats memStats;

  /// Output data collected per kernel name and port index.
  /// Key: "kernelName:portIdx"
  std::map<std::string, std::vector<uint64_t>> outputsByPort;
};

//===----------------------------------------------------------------------===//
// Core Specification (for building from components)
//===----------------------------------------------------------------------===//

/// Specification for adding a single core to the multi-core simulation.
struct CoreSpec {
  std::string name;
  std::string coreType;
  sim::StaticMappedModel model;
  std::vector<uint8_t> configBlob;
};

//===----------------------------------------------------------------------===//
// MultiCoreSimSession
//===----------------------------------------------------------------------===//

/// Top-level multi-core simulation orchestrator.
///
/// Manages multiple per-core CycleKernel instances, inter-core data transfers
/// via a NoC model, and system-level cycle accounting. The simulation runs in
/// lockstep: each global cycle advances all cores, then processes NoC transfers,
/// then handles memory hierarchy operations.
///
/// Usage:
///   1. Create session with configuration.
///   2. Add cores via addCore() and register inter-core routes.
///   3. Set per-core inputs.
///   4. Call run() to execute the simulation.
///   5. Retrieve results.
class MultiCoreSimSession {
public:
  explicit MultiCoreSimSession(const MultiCoreSimConfig &config = {});
  ~MultiCoreSimSession();

  MultiCoreSimSession(const MultiCoreSimSession &) = delete;
  MultiCoreSimSession &operator=(const MultiCoreSimSession &) = delete;

  // --- Build phase: add cores and configure routes ---

  /// Add a core to the simulation.
  /// Returns an error string on failure, empty string on success.
  std::string addCore(const CoreSpec &spec);

  /// Register an inter-core route for data transfer.
  std::string addInterCoreRoute(const InterCoreRoute &route);

  // --- Input ---

  /// Set input data for a specific core and port.
  std::string setCoreInput(const std::string &coreName, unsigned portIdx,
                           const std::vector<uint64_t> &data);

  /// Set external memory backing for a specific core.
  std::string setCoreExtMemory(const std::string &coreName, unsigned regionId,
                               uint8_t *data, size_t sizeBytes);

  // --- Execution ---

  /// Run the multi-core simulation to completion or timeout.
  MultiCoreSimResult run();

  // --- Output ---

  /// Get output data from a specific core and port.
  std::vector<uint64_t> getCoreOutput(const std::string &coreName,
                                      unsigned portIdx) const;

  // --- Callbacks ---

  /// Callback invoked after each global cycle.
  /// Arguments: (globalCycle, per-core states).
  using GlobalCycleCallback = std::function<void(
      uint64_t cycle, const std::vector<CoreSimResultEntry> &coreStates)>;

  void setGlobalCycleCallback(GlobalCycleCallback cb);

  // --- Queries ---

  unsigned getNumCores() const { return static_cast<unsigned>(cores_.size()); }

  const MultiCoreSimConfig &getConfig() const { return config_; }

private:
  /// Find a core wrapper by name. Returns nullptr if not found.
  CoreSimWrapper *findCore(const std::string &name);
  const CoreSimWrapper *findCore(const std::string &name) const;

  /// Advance all cores by one cycle.
  void stepAllCores();

  /// Process inter-core transfers: detect produced data, manage in-flight
  /// transfers, deliver completed transfers.
  void processInterCoreTransfers(uint64_t globalCycle);

  /// Process DMA requests from all cores through the memory hierarchy.
  void processMemoryHierarchy(uint64_t globalCycle);

  /// Check if all cores have completed execution.
  bool allCoresComplete() const;

  /// Collect per-core state snapshots for callback and result assembly.
  std::vector<CoreSimResultEntry> collectCoreStates(uint64_t globalCycle) const;

  /// Assemble the final simulation result from all cores and subsystems.
  MultiCoreSimResult assembleResult(uint64_t totalCycles) const;

  /// Compute link utilization statistics from accumulated link state.
  void computeLinkUtilization(uint64_t totalCycles);

  MultiCoreSimConfig config_;

  /// Per-core simulation wrappers.
  std::vector<std::unique_ptr<CoreSimWrapper>> cores_;

  /// Name to index mapping for cores.
  std::map<std::string, unsigned> coreNameToIdx_;

  /// Registered inter-core routes.
  std::vector<InterCoreRoute> routes_;

  /// In-flight transfers currently traversing the NoC.
  std::vector<PendingTransfer> inFlightTransfers_;

  /// NoC link states for utilization tracking.
  std::vector<NoCLinkState> linkStates_;

  /// Transfer statistics accumulated during simulation.
  TransferStats transferStats_;

  /// Memory statistics accumulated during simulation.
  MemSimStats memStats_;

  /// Global cycle callback.
  GlobalCycleCallback globalCycleCallback_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_MULTICORESIMSESSION_H

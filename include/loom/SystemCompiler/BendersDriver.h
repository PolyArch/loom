#ifndef LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H
#define LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace syscomp {

// Configuration for the Benders decomposition driver used by the system
// compiler to partition and schedule task graphs across multi-core fabrics.
struct BendersDriverOptions {
  // Maximum number of Benders iterations before giving up.
  unsigned maxIterations = 100;

  // Convergence tolerance: stop when the gap between upper and lower bounds
  // falls below this fraction.
  double convergenceTolerance = 1e-4;

  // Time limit in seconds for the entire Benders solve.
  double timeLimitSeconds = 300.0;

  // Number of cores available in the target fabric.
  unsigned numCores = 4;

  // Per-core SPM budget in bytes.
  uint64_t spmBudgetBytes = 65536;

  // NoC bandwidth in bytes per cycle.
  double nocBandwidthBytesPerCycle = 8.0;

  // Enable verbose logging of each iteration.
  bool verbose = false;
};

// Represents one task (kernel) in the task graph that Benders partitions.
struct BendersTask {
  std::string name;
  uint64_t estimatedCycles = 0;
  uint64_t spmBytes = 0;
  uint64_t outputBytes = 0;
};

// Represents a data dependency between two tasks.
struct BendersEdge {
  unsigned srcTaskIndex = 0;
  unsigned dstTaskIndex = 0;
  uint64_t dataBytes = 0;
};

// Result of the Benders decomposition: which task maps to which core.
struct BendersResult {
  bool feasible = false;
  std::string statusMessage;
  unsigned iterations = 0;
  double objectiveValue = 0.0;

  // taskAssignment[i] = core ID for task i.
  std::vector<unsigned> taskAssignment;
};

// Drives the Benders decomposition for system-level compilation.
class BendersDriver {
public:
  explicit BendersDriver(const BendersDriverOptions &options);

  void addTask(const BendersTask &task);
  void addEdge(const BendersEdge &edge);

  // Run the decomposition and return the result.
  BendersResult solve();

  const BendersDriverOptions &getOptions() const { return options_; }

private:
  BendersDriverOptions options_;
  std::vector<BendersTask> tasks_;
  std::vector<BendersEdge> edges_;
};

} // namespace syscomp
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H

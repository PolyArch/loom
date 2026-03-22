#ifndef LOOM_SIMULATOR_PORTTRACEEXPORTER_H
#define LOOM_SIMULATOR_PORTTRACEEXPORTER_H

#include "loom/Simulator/SimTypes.h"
#include "loom/Simulator/StaticModelTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace sim {

struct PortTraceEntry {
  uint64_t cycle = 0;
  bool valid = false;
  bool ready = false;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
  bool transferred = false;
};

// Records per-cycle port state for all ports of specified modules,
// then exports as hex trace files consumable by SV testbenches.
//
// Usage:
//   PortTraceExporter exporter(outputDir);
//   exporter.addTracedModule(moduleIndex, moduleName, portDescs);
//   // after each stepCycle():
//   exporter.recordCycle(cycle, portState);
//   // when done:
//   exporter.flush();
class PortTraceExporter {
public:
  explicit PortTraceExporter(const std::string &outputDir);

  struct TracedPort {
    unsigned portIndex;         // index into portState_
    StaticPortDirection dir;
    unsigned valueWidth;
    unsigned tagWidth;
    bool isTagged;
    std::string name;           // e.g. "in0", "out1"
  };

  struct TracedModule {
    unsigned moduleIndex;
    std::string moduleName;
    std::vector<TracedPort> ports;
  };

  void addTracedModule(unsigned moduleIndex,
                       const std::string &moduleName,
                       const std::vector<TracedPort> &ports);

  // Call after each stepCycle(). portState is the kernel's portState_.
  void recordCycle(uint64_t cycle,
                   const std::vector<SimChannel> &portState);

  // Write all collected traces to output files.
  // Returns true on success.
  bool flush() const;

private:
  std::string outputDir_;
  std::vector<TracedModule> tracedModules_;

  // Per-port, per-cycle trace data.
  // Indexed: [module_idx_in_tracedModules][port_idx_in_module][cycle_seq]
  std::vector<std::vector<std::vector<PortTraceEntry>>> traceData_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_PORTTRACEEXPORTER_H

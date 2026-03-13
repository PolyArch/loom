//===-- SimArtifactWriter.h - Trace/stat output serialization ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Serializes simulation trace events to .trace files and performance statistics
// to .stat JSON files. Per spec-cosim-backend-eventsim.md, output filenames
// follow the <dfg>_on_<adg>.trace / .stat convention.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMARTIFACTWRITER_H
#define LOOM_SIMULATOR_SIMARTIFACTWRITER_H

#include "loom/Simulator/SimTypes.h"

#include <string>
#include <vector>

namespace loom {
namespace sim {

struct SimResult; // Forward declaration.

/// Host-side timing breakdown for the stat file.
struct HostTiming {
  double configSeconds = 0.0;
  double hostExecSeconds = 0.0;
  double accelExecSeconds = 0.0;
  double totalSeconds = 0.0;
};

/// Write trace events to a binary .trace file.
/// Returns true on success.
bool writeTraceFile(const std::string &path,
                    const std::vector<TraceEvent> &events);

/// Write performance statistics to a JSON .stat file.
/// Returns true on success.
bool writeStatFile(const std::string &path, const SimResult &result,
                   const HostTiming &timing = HostTiming());

/// Apply event-kind filter to trace events.
/// Keeps only events whose kind is in the allowedKinds bitmask.
/// Bit N corresponds to EventKind N.
std::vector<TraceEvent>
filterByEventKind(const std::vector<TraceEvent> &events, uint16_t allowedKinds);

/// Apply node filter to trace events.
/// Keeps only events whose hwNodeId is in the allowedNodes set.
/// Events with hwNodeId=0 (session-level events) are always kept.
std::vector<TraceEvent>
filterByNode(const std::vector<TraceEvent> &events,
             const std::vector<uint32_t> &allowedNodes);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMARTIFACTWRITER_H

//===-- SimArtifactWriter.cpp - Trace/stat serialization -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimArtifactWriter.h"
#include "loom/Simulator/SimEngine.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace loom {
namespace sim {

//===----------------------------------------------------------------------===//
// Trace file writer (binary format)
//===----------------------------------------------------------------------===//

/// Binary trace format:
///   Header: 4-byte magic "LTRC", 4-byte version (1), 8-byte event count
///   Events: packed LoomTraceEvent records (38 bytes each)
///     uint64_t cycle
///     uint32_t epochId
///     uint64_t invocationId
///     uint16_t coreId
///     uint32_t hwNodeId
///     uint8_t  eventKind
///     uint8_t  lane
///     uint16_t flags
///     uint32_t arg0
///     uint32_t arg1

bool writeTraceFile(const std::string &path,
                    const std::vector<TraceEvent> &events) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;

  // Header.
  const char magic[4] = {'L', 'T', 'R', 'C'};
  uint32_t version = 1;
  uint64_t count = events.size();
  out.write(magic, 4);
  out.write(reinterpret_cast<const char *>(&version), 4);
  out.write(reinterpret_cast<const char *>(&count), 8);

  // Events.
  for (const auto &ev : events) {
    out.write(reinterpret_cast<const char *>(&ev.cycle), 8);
    out.write(reinterpret_cast<const char *>(&ev.epochId), 4);
    out.write(reinterpret_cast<const char *>(&ev.invocationId), 8);
    out.write(reinterpret_cast<const char *>(&ev.coreId), 2);
    out.write(reinterpret_cast<const char *>(&ev.hwNodeId), 4);
    out.write(reinterpret_cast<const char *>(&ev.eventKind), 1);
    out.write(reinterpret_cast<const char *>(&ev.lane), 1);
    out.write(reinterpret_cast<const char *>(&ev.flags), 2);
    out.write(reinterpret_cast<const char *>(&ev.arg0), 4);
    out.write(reinterpret_cast<const char *>(&ev.arg1), 4);
  }

  return out.good();
}

//===----------------------------------------------------------------------===//
// Stat file writer (JSON format)
//===----------------------------------------------------------------------===//

static std::string escapeJson(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
    }
  }
  return out;
}

bool writeStatFile(const std::string &path, const SimResult &result,
                   const HostTiming &timing) {
  std::ofstream out(path);
  if (!out)
    return false;

  out << "{\n";
  out << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
  out << "  \"totalCycles\": " << result.totalCycles << ",\n";
  out << "  \"configCycles\": " << result.configCycles << ",\n";
  out << "  \"executionCycles\": "
      << (result.totalCycles > result.configCycles
              ? result.totalCycles - result.configCycles
              : 0)
      << ",\n";

  if (!result.errorMessage.empty())
    out << "  \"errorMessage\": \"" << escapeJson(result.errorMessage)
        << "\",\n";

  // Host timing breakdown.
  out << "  \"hostTiming\": {\n";
  out << "    \"host_config_time\": " << timing.configSeconds << ",\n";
  out << "    \"host_exec_time\": " << timing.hostExecSeconds << ",\n";
  out << "    \"accel_exec_time\": " << timing.accelExecSeconds << ",\n";
  out << "    \"total_time\": " << timing.totalSeconds << "\n";
  out << "  },\n";

  // Per-node performance with derived metrics.
  out << "  \"nodePerf\": [\n";
  for (size_t i = 0; i < result.nodePerf.size(); ++i) {
    const auto &p = result.nodePerf[i];
    uint64_t nodeTotalCycles = p.activeCycles + p.stallCyclesIn + p.stallCyclesOut;
    double utilization =
        (nodeTotalCycles > 0)
            ? static_cast<double>(p.activeCycles) / nodeTotalCycles
            : 0.0;
    double inputStallRatio =
        (nodeTotalCycles > 0)
            ? static_cast<double>(p.stallCyclesIn) / nodeTotalCycles
            : 0.0;
    double outputStallRatio =
        (nodeTotalCycles > 0)
            ? static_cast<double>(p.stallCyclesOut) / nodeTotalCycles
            : 0.0;
    double throughputProxy =
        (p.activeCycles > 0)
            ? static_cast<double>(p.tokensOut) / p.activeCycles
            : 0.0;

    out << "    {\n";
    out << "      \"nodeIndex\": " << i << ",\n";
    out << "      \"activeCycles\": " << p.activeCycles << ",\n";
    out << "      \"stallCyclesIn\": " << p.stallCyclesIn << ",\n";
    out << "      \"stallCyclesOut\": " << p.stallCyclesOut << ",\n";
    out << "      \"tokensIn\": " << p.tokensIn << ",\n";
    out << "      \"tokensOut\": " << p.tokensOut << ",\n";
    out << "      \"configWrites\": " << p.configWrites << ",\n";
    out << "      \"utilization\": " << utilization << ",\n";
    out << "      \"inputStallRatio\": " << inputStallRatio << ",\n";
    out << "      \"outputStallRatio\": " << outputStallRatio << ",\n";
    out << "      \"throughputProxy\": " << throughputProxy << "\n";
    out << "    }";
    if (i + 1 < result.nodePerf.size())
      out << ",";
    out << "\n";
  }
  out << "  ],\n";

  // Summary derived metrics.
  uint64_t totalActive = 0, totalStallIn = 0, totalStallOut = 0;
  uint64_t totalTokensIn = 0, totalTokensOut = 0;
  for (const auto &p : result.nodePerf) {
    totalActive += p.activeCycles;
    totalStallIn += p.stallCyclesIn;
    totalStallOut += p.stallCyclesOut;
    totalTokensIn += p.tokensIn;
    totalTokensOut += p.tokensOut;
  }

  uint64_t execCycles =
      (result.totalCycles > result.configCycles)
          ? result.totalCycles - result.configCycles
          : 0;
  double configOverheadRatio =
      (result.totalCycles > 0)
          ? static_cast<double>(result.totalConfigWrites) / result.totalCycles
          : 0.0;

  out << "  \"summary\": {\n";
  out << "    \"nodeCount\": " << result.nodePerf.size() << ",\n";
  out << "    \"totalActiveCycles\": " << totalActive << ",\n";
  out << "    \"totalStallInCycles\": " << totalStallIn << ",\n";
  out << "    \"totalStallOutCycles\": " << totalStallOut << ",\n";
  out << "    \"totalTokensIn\": " << totalTokensIn << ",\n";
  out << "    \"totalTokensOut\": " << totalTokensOut << ",\n";
  out << "    \"totalConfigWrites\": " << result.totalConfigWrites << ",\n";
  out << "    \"configOverheadRatio\": " << configOverheadRatio << ",\n";
  out << "    \"traceEventCount\": " << result.traceEvents.size() << "\n";
  out << "  }\n";

  out << "}\n";

  return out.good();
}

//===----------------------------------------------------------------------===//
// Event filtering
//===----------------------------------------------------------------------===//

std::vector<TraceEvent>
filterByEventKind(const std::vector<TraceEvent> &events,
                  uint16_t allowedKinds) {
  std::vector<TraceEvent> filtered;
  filtered.reserve(events.size());
  for (const auto &ev : events) {
    if ((1u << ev.eventKind) & allowedKinds)
      filtered.push_back(ev);
  }
  return filtered;
}

std::vector<TraceEvent>
filterByNode(const std::vector<TraceEvent> &events,
             const std::vector<uint32_t> &allowedNodes) {
  std::unordered_set<uint32_t> nodeSet(allowedNodes.begin(),
                                        allowedNodes.end());
  std::vector<TraceEvent> filtered;
  filtered.reserve(events.size());
  for (const auto &ev : events) {
    // Session-level events (invocation start/done, config write with
    // hwNodeId=0) are always kept.
    if (ev.hwNodeId == 0 || nodeSet.count(ev.hwNodeId))
      filtered.push_back(ev);
  }
  return filtered;
}

} // namespace sim
} // namespace loom

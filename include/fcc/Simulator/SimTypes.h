#ifndef FCC_SIMULATOR_SIMTYPES_H
#define FCC_SIMULATOR_SIMTYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace fcc {
namespace sim {

struct SimChannel {
  bool valid = false;
  bool ready = false;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;

  bool transferred() const { return valid && ready; }

  void clearForward() {
    valid = false;
    data = 0;
    tag = 0;
  }
};

enum class EventKind : uint8_t {
  NodeFire = 0,
  NodeStallIn = 1,
  NodeStallOut = 2,
  RouteUse = 3,
  ConfigWrite = 4,
  InvocationStart = 5,
  InvocationDone = 6,
  DeviceError = 7,
};

struct TraceEvent {
  uint64_t cycle = 0;
  uint32_t epochId = 0;
  uint64_t invocationId = 0;
  uint16_t coreId = 0;
  uint32_t hwNodeId = 0;
  EventKind eventKind = EventKind::NodeFire;
  uint8_t lane = 0;
  uint16_t flags = 0;
  uint32_t arg0 = 0;
  uint32_t arg1 = 0;
};

struct PerfSnapshot {
  uint32_t nodeIndex = 0;
  uint64_t activeCycles = 0;
  uint64_t stallCyclesIn = 0;
  uint64_t stallCyclesOut = 0;
  uint64_t tokensIn = 0;
  uint64_t tokensOut = 0;
  uint64_t configWrites = 0;
};

enum class TraceMode : uint8_t {
  Off = 0,
  Summary = 1,
  Full = 2,
};

enum class RunTermination : uint8_t {
  Completed = 0,
  Timeout = 1,
  DeviceError = 2,
  ContractError = 3,
};

struct SimConfig {
  uint32_t configWordsPerCycle = 1;
  uint32_t resetOverheadCycles = 1;
  uint32_t extMemLatency = 10;
  TraceMode traceMode = TraceMode::Full;
  uint64_t maxCycles = 1000000;
  uint16_t coreId = 0;
  std::vector<EventKind> traceFilterKinds;
  std::vector<uint32_t> traceFilterNodes;
  std::vector<uint16_t> traceFilterCores;
};

struct SimResult {
  bool success = false;
  RunTermination termination = RunTermination::ContractError;
  uint64_t totalCycles = 0;
  uint64_t configCycles = 0;
  uint64_t totalConfigWrites = 0;
  std::string errorMessage;
  std::vector<PerfSnapshot> nodePerf;
  std::vector<TraceEvent> traceEvents;
};

const char *eventKindName(EventKind kind);
const char *runTerminationName(RunTermination termination);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMTYPES_H

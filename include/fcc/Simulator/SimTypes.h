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
  uint64_t generation = 0;
  bool didTransfer = false;

  bool transferred() const { return didTransfer; }

  void clearForward() {
    valid = false;
    data = 0;
    tag = 0;
    hasTag = false;
    generation = 0;
    didTransfer = false;
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

enum class SimPhase : uint8_t {
  Evaluate = 0,
  Commit = 1,
};

struct TraceEvent {
  uint64_t cycle = 0;
  SimPhase phase = SimPhase::Commit;
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

enum class BoundaryReason : uint8_t {
  NeedMemIssue = 0,
  WaitMemResp = 1,
  InvocationDone = 2,
  Deadlock = 3,
  BudgetHit = 4,
  None = 255,
};

struct TraceModuleInfo {
  uint32_t hwNodeId = 0;
  std::string kind;
  std::string name;
  std::string componentName;
  std::string functionUnitName;
  int32_t boundaryOrdinal = -1;
};

struct TraceDocument {
  uint32_t version = 1;
  std::string traceKind = "fcc_cycle_trace";
  std::string producer = "fcc";
  uint32_t epochId = 0;
  uint64_t invocationId = 0;
  uint16_t coreId = 0;
  std::vector<TraceModuleInfo> modules;
  std::vector<TraceEvent> events;
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

struct NamedCounter {
  std::string name;
  uint64_t value = 0;
};

struct FinalStatePortSnapshot {
  uint32_t portId = 0;
  uint32_t parentNodeId = 0;
  bool isInput = false;
  bool valid = false;
  bool ready = false;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
  uint64_t generation = 0;
};

struct FinalStateEdgeSnapshot {
  uint32_t edgeIndex = 0;
  uint32_t hwEdgeId = 0;
  uint32_t srcPort = 0;
  uint32_t dstPort = 0;
  bool valid = false;
  bool ready = false;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
  uint64_t generation = 0;
};

struct FinalStateModuleSnapshot {
  uint32_t hwNodeId = 0;
  std::string name;
  std::string kind;
  bool hasPendingWork = false;
  uint64_t collectedTokenCount = 0;
  uint64_t logicalFireCount = 0;
  uint64_t inputCaptureCount = 0;
  uint64_t outputTransferCount = 0;
  std::string debugState;
  std::vector<NamedCounter> counters;
};

struct FinalStateSummary {
  bool obligationsSatisfied = false;
  bool quiescent = false;
  bool done = false;
  bool deadlocked = false;
  uint64_t idleCycleStreak = 0;
  uint64_t outstandingMemoryRequestCount = 0;
  uint64_t completedMemoryResponseCount = 0;
  std::vector<FinalStatePortSnapshot> livePorts;
  std::vector<FinalStateEdgeSnapshot> liveEdges;
  std::vector<FinalStateModuleSnapshot> pendingModules;
  std::vector<FinalStateModuleSnapshot> moduleSummaries;
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
  TraceDocument traceDocument;
  std::vector<TraceEvent> traceEvents;
  FinalStateSummary finalState;
};

const char *eventKindName(EventKind kind);
const char *runTerminationName(RunTermination termination);
const char *simPhaseName(SimPhase phase);
const char *boundaryReasonName(BoundaryReason reason);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMTYPES_H

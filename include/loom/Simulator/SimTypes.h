#ifndef LOOM_SIMULATOR_SIMTYPES_H
#define LOOM_SIMULATOR_SIMTYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
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
  std::string traceKind = "loom_cycle_trace";
  std::string producer = "loom";
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

struct ConfigLoadSummary {
  uint64_t wordCount = 0;
  uint64_t byteCount = 0;
  uint64_t wordsPerCycle = 0;
  uint64_t startCycle = 0;
  uint64_t endCycle = 0;
  uint64_t cycles = 0;
  uint64_t dmaRequestCount = 0;
  uint64_t dmaReadBytes = 0;
  uint64_t dmaStartTick = 0;
  uint64_t dmaEndTick = 0;
  uint64_t dmaElapsedTicks = 0;
  uint64_t kernelLaunchCycle = 0;
  uint64_t kernelFirstActiveCycle = 0;
  uint64_t configExecOverlapCycles = 0;
  uint64_t configExecExposedCycles = 0;
  double configOverlapEfficiency = 0.0;
};

struct ConfigSliceTiming {
  std::string name;
  std::string kind;
  uint32_t hwNodeId = 0;
  uint32_t wordOffset = 0;
  uint32_t wordCount = 0;
  uint64_t startCycle = 0;
  uint64_t endCycle = 0;
};

struct NamedCounter {
  std::string name;
  uint64_t value = 0;
};

struct ModulePerfDetail {
  uint32_t hwNodeId = 0;
  std::string name;
  std::string kind;
  std::string componentName;
  std::string functionUnitName;
  bool configured = false;
  bool staticallyUsed = false;
  bool dynamicallyUsed = false;
  uint64_t configReadyCycle = 0;
  bool hasFirstUseCycle = false;
  uint64_t firstUseCycle = 0;
  int64_t configSlackCycles = 0;
  uint64_t activeCycles = 0;
  uint64_t stallCyclesIn = 0;
  uint64_t stallCyclesOut = 0;
  uint64_t tokensIn = 0;
  uint64_t tokensOut = 0;
  uint64_t logicalFireCount = 0;
  uint64_t inputCaptureCount = 0;
  uint64_t outputTransferCount = 0;
  uint64_t outputBusyCycles = 0;
  uint64_t inputLatchedCycles = 0;
  double dynamicUtilization = 0.0;
  std::vector<NamedCounter> counters;
};

struct MemoryRegionPerfSummary {
  uint32_t regionId = 0;
  int32_t slot = -1;
  uint64_t loadRequestCount = 0;
  uint64_t storeRequestCount = 0;
  uint64_t loadBytes = 0;
  uint64_t storeBytes = 0;
  bool hasFirstRequestCycle = false;
  uint64_t firstRequestCycle = 0;
  bool hasLastCompletionCycle = false;
  uint64_t lastCompletionCycle = 0;
};

struct FabricStaticUtilizationSummary {
  uint64_t totalModules = 0;
  uint64_t configuredModules = 0;
  uint64_t totalFunctionUnits = 0;
  uint64_t mappedFunctionUnits = 0;
  uint64_t totalSpatialPEs = 0;
  uint64_t usedSpatialPEs = 0;
  uint64_t totalTemporalPEs = 0;
  uint64_t usedTemporalPEs = 0;
  double configuredModuleRatio = 0.0;
  double mappedFunctionUnitRatio = 0.0;
  double usedSpatialPERatio = 0.0;
  double usedTemporalPERatio = 0.0;
};

struct FabricDynamicUtilizationSummary {
  uint64_t kernelCycles = 0;
  uint64_t activeCycles = 0;
  uint64_t idleCycles = 0;
  uint64_t fabricActiveCycles = 0;
  uint64_t needMemIssueCycles = 0;
  uint64_t waitMemRespCycles = 0;
  uint64_t budgetBoundaryCount = 0;
  uint64_t deadlockBoundaryCount = 0;
  uint64_t maxInflightMemoryRequests = 0;
  double activeCycleRatio = 0.0;
  double fabricActiveRatio = 0.0;
  double memIssueRatio = 0.0;
  double memWaitRatio = 0.0;
};

struct AcceleratorStats {
  uint64_t totalCycles = 0;
  uint64_t kernelCycles = 0;
  uint64_t deviceElapsedTicks = 0;
  uint64_t memoryIoTicks = 0;
  uint64_t loadRequestCount = 0;
  uint64_t storeRequestCount = 0;
  uint64_t loadBytes = 0;
  uint64_t storeBytes = 0;
  ConfigLoadSummary configLoad;
  FabricStaticUtilizationSummary staticUtilization;
  FabricDynamicUtilizationSummary dynamicUtilization;
  std::vector<ConfigSliceTiming> configSlices;
  std::vector<ModulePerfDetail> modules;
  std::vector<MemoryRegionPerfSummary> memoryRegions;
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
  bool hardwareEmpty = false;
  bool quiescent = false;
  bool done = false;
  bool deadlocked = false;
  uint64_t idleCycleStreak = 0;
  uint64_t outstandingMemoryRequestCount = 0;
  uint64_t completedMemoryResponseCount = 0;
  std::string terminationAuditError;
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
  AcceleratorStats acceleratorStats;
  TraceDocument traceDocument;
  std::vector<TraceEvent> traceEvents;
  FinalStateSummary finalState;
};

const char *eventKindName(EventKind kind);
const char *runTerminationName(RunTermination termination);
const char *simPhaseName(SimPhase phase);
const char *boundaryReasonName(BoundaryReason reason);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTYPES_H

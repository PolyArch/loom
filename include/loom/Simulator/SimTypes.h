//===-- SimTypes.h - Simulator core data types -------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Core data types for the event-driven cycle-accurate fabric simulator.
// Defines channels, trace events, and performance snapshots.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMTYPES_H
#define LOOM_SIMULATOR_SIMTYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace sim {

//===----------------------------------------------------------------------===//
// SimChannel - valid/ready handshake channel between modules
//===----------------------------------------------------------------------===//

/// A unidirectional data channel connecting a producer output to a consumer
/// input. Forward signals (valid, data, tag) flow from producer to consumer.
/// The ready signal flows from consumer to producer.
struct SimChannel {
  bool valid = false;
  bool ready = false;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;

  /// True when a handshake transfer occurs this cycle.
  bool transferred() const { return valid && ready; }

  /// Clear forward signals (valid/data/tag). Called between evaluation rounds.
  void clearForward() {
    valid = false;
    data = 0;
    tag = 0;
  }

  /// Snapshot for fixed-point convergence detection.
  bool operator==(const SimChannel &o) const {
    return valid == o.valid && ready == o.ready && data == o.data &&
           tag == o.tag && hasTag == o.hasTag;
  }
  bool operator!=(const SimChannel &o) const { return !(*this == o); }
};

//===----------------------------------------------------------------------===//
// Event kinds (per spec-cosim-trace.md)
//===----------------------------------------------------------------------===//

enum EventKind : uint8_t {
  EV_NODE_FIRE = 0,
  EV_NODE_STALL_IN = 1,
  EV_NODE_STALL_OUT = 2,
  EV_ROUTE_USE = 3,
  EV_CONFIG_WRITE = 4,
  EV_INVOCATION_START = 5,
  EV_INVOCATION_DONE = 6,
  EV_DEVICE_ERROR = 7,
};

//===----------------------------------------------------------------------===//
// TraceEvent (per spec-cosim-trace.md LoomTraceEvent)
//===----------------------------------------------------------------------===//

struct TraceEvent {
  uint64_t cycle = 0;
  uint32_t epochId = 0;
  uint64_t invocationId = 0;
  uint16_t coreId = 0;
  uint32_t hwNodeId = 0;
  EventKind eventKind = EV_NODE_FIRE;
  uint8_t lane = 0;
  uint16_t flags = 0;
  uint32_t arg0 = 0;
  uint32_t arg1 = 0;
};

//===----------------------------------------------------------------------===//
// PerfSnapshot (per spec-cosim-trace.md LoomPerfSnapshot)
//===----------------------------------------------------------------------===//

struct PerfSnapshot {
  uint32_t nodeIndex = 0; // hwNodeId from the ADG graph.
  uint64_t activeCycles = 0;
  uint64_t stallCyclesIn = 0;
  uint64_t stallCyclesOut = 0;
  uint64_t tokensIn = 0;
  uint64_t tokensOut = 0;
  uint64_t configWrites = 0;
};

//===----------------------------------------------------------------------===//
// Runtime error codes (per spec-fabric-error.md)
//===----------------------------------------------------------------------===//

namespace RtError {
inline constexpr uint16_t OK = 0;
// CFG errors (1-15)
inline constexpr uint16_t CFG_SWITCH_ROUTE_MIX = 1;
inline constexpr uint16_t CFG_ADG_COMBINATIONAL_LOOP = 2;
inline constexpr uint16_t CFG_TEMPORAL_SW_SAME_TAG_INPUTS = 4;
inline constexpr uint16_t CFG_TEMPORAL_SW_DUP_TAG = 5;
inline constexpr uint16_t CFG_TEMPORAL_PE_DUP_TAG = 6;
inline constexpr uint16_t CFG_TEMPORAL_PE_ILLEGAL_REG = 7;
inline constexpr uint16_t CFG_TEMPORAL_PE_REG_TAG_NONZERO = 8;
inline constexpr uint16_t CFG_MAP_TAG_DUP_TAG = 9;
inline constexpr uint16_t CFG_PE_STREAM_CONT_COND_ONEHOT = 10;
inline constexpr uint16_t CFG_PE_CMPI_PREDICATE_INVALID = 11;
inline constexpr uint16_t CFG_MEMORY_OVERLAP_TAG_REGION = 12;
inline constexpr uint16_t CFG_MEMORY_EMPTY_TAG_RANGE = 13;
inline constexpr uint16_t CFG_EXTMEMORY_OVERLAP_TAG_REGION = 14;
inline constexpr uint16_t CFG_EXTMEMORY_EMPTY_TAG_RANGE = 15;
// RT errors (256+)
inline constexpr uint16_t RT_TEMPORAL_PE_NO_MATCH = 256;
inline constexpr uint16_t RT_TEMPORAL_SW_NO_MATCH = 257;
inline constexpr uint16_t RT_MAP_TAG_NO_MATCH = 258;
inline constexpr uint16_t RT_DATAFLOW_STREAM_ZERO_STEP = 259;
inline constexpr uint16_t RT_MEMORY_TAG_OOB = 260;
inline constexpr uint16_t RT_MEMORY_STORE_DEADLOCK = 261;
inline constexpr uint16_t RT_SWITCH_UNROUTED_INPUT = 262;
inline constexpr uint16_t RT_TEMPORAL_SW_UNROUTED_INPUT = 263;
inline constexpr uint16_t RT_MEMORY_NO_MATCH = 264;
inline constexpr uint16_t RT_EXTMEMORY_NO_MATCH = 265;
} // namespace RtError

//===----------------------------------------------------------------------===//
// Trace collection mode
//===----------------------------------------------------------------------===//

enum class TraceMode : uint8_t {
  Off = 0,
  Summary = 1,
  Full = 2,
};

//===----------------------------------------------------------------------===//
// Simulation configuration
//===----------------------------------------------------------------------===//

struct SimConfig {
  /// Config_mem programming rate: words per cycle.
  uint32_t configWordsPerCycle = 1;
  /// Reset overhead in cycles after config programming.
  uint32_t resetOverheadCycles = 1;
  /// External memory latency in cycles.
  uint32_t extMemLatency = 10;
  /// Trace collection mode.
  TraceMode traceMode = TraceMode::Full;
  /// Maximum cycles before timeout (0 = no limit).
  uint64_t maxCycles = 1000000;

  /// Trace filters (empty = include all).
  /// When non-empty, only events matching the filter are collected.
  std::vector<EventKind> traceFilterKinds;  // event-kind whitelist
  std::vector<uint32_t> traceFilterNodes;   // hwNodeId whitelist
  std::vector<uint16_t> traceFilterCores;   // coreId whitelist
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTYPES_H

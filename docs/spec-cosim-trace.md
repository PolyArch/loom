# Loom Co-Simulation Trace and Performance Specification

## Overview

This document is the authoritative specification for trace events and
performance statistics produced during Loom `cosim` execution.

It defines data schemas, event semantics, ordering rules, and derived metrics.
Protocol transport for telemetry is defined in
[spec-cosim-protocol.md](./spec-cosim-protocol.md).

## Goals

Trace/perf collection must enable:

- per-node execution visibility
- per-core activity visibility in heterogeneous accelerator models
- cycle-level debugging and regression comparison
- low-overhead summary statistics for performance evaluation

## Identifier Model

Telemetry records use these identifiers:

- `epoch_id`: configuration epoch in current session
- `invocation_id`: host invocation identifier
- `core_id`: logical accelerator core identifier
- `hw_node_id`: hardware node identifier

`hw_node_id` mapping to Fabric/ADG entities must be stable within one backend
build. The producer of this mapping is backend/export metadata, not this
document.

## Event Record Schema

### Canonical event structure

```c
struct LoomTraceEvent {
  uint64_t cycle;
  uint32_t epoch_id;
  uint64_t invocation_id;
  uint16_t core_id;
  uint32_t hw_node_id;
  uint8_t  event_kind;
  uint8_t  lane;
  uint16_t flags;
  uint32_t arg0;
  uint32_t arg1;
};
```

Field semantics:

- `cycle`: global simulation cycle timestamp
- `lane`: optional sub-port/lane index
- `flags`: backend-defined bit flags with documented meaning
- `arg0/arg1`: event-specific payload

### Event kind enumeration

Minimum required event kinds:

| Value | Name | Meaning |
|------|------|---------|
| `0` | `EV_NODE_FIRE` | node accepted inputs and produced output action |
| `1` | `EV_NODE_STALL_IN` | blocked waiting for input |
| `2` | `EV_NODE_STALL_OUT` | blocked waiting for output ready |
| `3` | `EV_ROUTE_USE` | routed edge/path segment active |
| `4` | `EV_CONFIG_WRITE` | config word observed/committed |
| `5` | `EV_INVOCATION_START` | invocation entered running state |
| `6` | `EV_INVOCATION_DONE` | invocation completed |
| `7` | `EV_DEVICE_ERROR` | backend reported runtime error |

Implementations may add more values but must preserve these base meanings.

## Performance Statistics Schema

### Canonical snapshot structure

```c
struct LoomPerfSnapshot {
  uint64_t cycle;
  uint32_t epoch_id;
  uint64_t invocation_id;
  uint16_t core_id;
  uint64_t active_cycles;
  uint64_t stall_cycles_in;
  uint64_t stall_cycles_out;
  uint64_t tokens_in;
  uint64_t tokens_out;
  uint64_t config_writes;
};
```

Snapshots may be emitted periodically or only at invocation end.

## Collection Modes

Implementations must support at least one of the following modes:

- **Off**: no trace/perf collection
- **Summary**: only performance snapshots/counters
- **Full trace**: all event records plus summary counters

Mode is selected by runtime options before invocation start.

## Ordering Rules

- events are globally ordered by `(cycle, sequence_in_cycle)`
- same-cycle ordering must be deterministic for deterministic mode
- `EV_INVOCATION_START` precedes all node events of same invocation
- `EV_INVOCATION_DONE` follows all functional output-producing events

## Transport and Storage

Telemetry may be delivered through:

- typed ESI channels (online)
- sidecar files (offline)

If both are enabled:

- channel stream is authoritative for online control decisions
- sidecar files must be a lossless projection of emitted stream or explicitly
  declare sampling/dropping policy

Recommended sidecar artifacts:

- `<run>.trace.bin` (event records)
- `<run>.perf.json` (summary metrics)

## Derived Metrics

The following derived metrics are standardized:

- node utilization: `active_cycles / total_cycles`
- input-stall ratio: `stall_cycles_in / total_cycles`
- output-stall ratio: `stall_cycles_out / total_cycles`
- throughput proxy: `tokens_out / active_cycles` (when active_cycles > 0)
- config overhead: `config_writes / total_cycles`

`total_cycles` is the cycle span of the measured window.

## Overhead Control

Trace/perf instrumentation must expose controls for runtime overhead:

- event filtering by event kind
- node/core filters
- sampling ratio (optional)
- bounded buffering policy

When dropping is enabled, drop counts must be reported explicitly.

## Validation Requirements

A trace/perf implementation is valid when:

- schema fields are populated according to this document
- ordering rules are satisfied
- base event kinds are emitted correctly
- derived metrics match raw counters within arithmetic consistency

End-to-end checks are defined in [spec-cosim-validation.md](./spec-cosim-validation.md).

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
- [spec-viz-hw.md](./spec-viz-hw.md)

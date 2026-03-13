# Loom Co-Simulation Event-Driven Simulator Backend Specification

## Overview

This document defines how Loom `cosim` binds to an in-process event-driven
cycle-accurate simulator backend. This backend is a lightweight alternative to
the SystemC and RTL backends: it evaluates mapped DFGs on configured ADGs
entirely in C++ without requiring SystemC kernels, RTL simulators, or ESI
transport.

General protocol and runtime contracts are defined in:

- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)

Architecture and lifecycle contracts are defined in
[spec-cosim-architecture.md](./spec-cosim-architecture.md).

## Backend Role

The event-driven simulator backend provides:

- one logical simulation thread with global cycle progression (single-threaded
  in-process execution; no separate process or RPC)
- direct C++ `config_mem` programming path (MMIO-style word writes via
  in-process API; no ESI transport required)
- application data channel path (typed C++ vectors for input/output payloads)
- trace/perf collection path (in-memory event buffers and statistics snapshots)
- CPU oracle comparison path (host reference execution and output comparison)

## Required Inputs

The event-driven backend consumes:

- flattened ADG graph (`Graph` from `ADGFlattener`)
- mapper-produced `config_mem` binary blob (`config.bin`)
- host input vectors (per-port data and optional tags)
- host reference outputs for CPU oracle comparison

## Architecture

### In-Process Model

Unlike SystemC and RTL backends that may run as separate processes with ESI RPC
transport, the event-driven backend runs entirely in-process:

- `EventSimSession` owns a `SimEngine` instance
- No network transport; all calls are direct C++ method invocations
- Thread safety: single-threaded dispatcher model; the public API is
  thread-safe at the session boundary but internally serialized

### Component Hierarchy

```
EventSimSession (session lifecycle, state machine)
  +-- SimEngine (two-phase evaluation loop, module graph)
       +-- SimModule[] (fabric module models)
       +-- SimChannel[] (inter-module channels)
```

## Session Lifecycle

The session follows the state machine defined in
[spec-cosim-architecture.md](./spec-cosim-architecture.md):

| State | Entry | Exit |
|-------|-------|------|
| `Created` | constructor | `connect()` |
| `Connected` | `connect()` validates engine readiness | ADG loaded and validated |
| `Ready` | ADG graph built and module topology verified | `loadConfig()` |
| `Configured` | `config_mem` image applied to all modules | `invoke()` |
| `Running` | simulation loop active | completion or fatal error |
| `Draining` | all inputs consumed, collecting remaining outputs | outputs drained |
| `Verified` | CPU oracle comparison completed | next epoch or `disconnect()` |
| `Closed` | `disconnect()` or fatal error | terminal |

### State Transition Rules

- `connect()`: `Created` -> `Connected`. Validates that the `SimEngine` can
  accept an ADG graph.
- `buildFromGraph()`: `Connected` -> `Ready`. Builds module topology, computes
  topological order, detects combinational cycles.
- `loadConfig()`: `Ready` or `Verified` -> `Configured`. Programs `config_mem`
  words to module models. Rejected in `Running` state.
- `invoke()`: `Configured` -> `Running` -> `Draining`. Executes the simulation
  loop until completion or timeout.
- `compare()`: `Draining` -> `Verified`. Runs CPU oracle comparison.
- `disconnect()`: any state -> `Closed`. Releases resources.

Invalid transitions return an error status with the current and requested states.

### Multi-Epoch Support

The backend supports multiple configuration epochs within one session:

1. After `Verified`, the session may transition back to `Configured` via a new
   `loadConfig()` call (reconfiguration).
2. `resetExecution()` clears runtime state but preserves configuration (for
   repeated invocation with same config).
3. `resetAll()` clears both runtime state and configuration (for full
   reconfiguration).

Each epoch increments the `epoch_id` counter used in trace events.

## Configuration Path

### config_mem Programming

The backend accepts configuration via two paths:

1. **File path**: `loadConfig(const std::string &path)` reads a `config.bin`
   binary file.
2. **Raw bytes**: `loadConfig(const std::vector<uint8_t> &blob)` accepts
   in-memory configuration bytes.

Config words are distributed to modules via `moduleConfigMap_` (word offset and
count per module). Each module's `configure()` method receives its slice of
config words.

### Configuration Overhead Modeling

Configuration programming is modeled with configurable overhead:

- `config_rate`: words programmed per cycle (default: 1)
- `reset_overhead`: cycles for post-config reset (default: 0)
- Config cycles = ceil(total_words / config_rate) + reset_overhead
- Each config word emits an `EV_CONFIG_WRITE` trace event

## Simulation Thread Contract

The event-driven backend uses a two-phase per-cycle evaluation model:

### Phase 1: Combinational Convergence

Evaluates all combinational modules in topological order. If combinational
cycles exist (detected by Kahn's algorithm), uses fixed-point iteration until
signals stabilize or iteration limit is reached.

Combinational modules: `fabric.switch`, `fabric.add_tag`, `fabric.map_tag`,
`fabric.del_tag`, bypassed `fabric.fifo`, `fabric.pe`.

### Phase 2: Sequential State Advance

Advances clock for all sequential modules. Consumes transferred tokens
(valid && ready handshake) and updates internal state.

Sequential modules: `fabric.temporal_pe`, `fabric.temporal_sw`, non-bypassed
`fabric.fifo`, `fabric.memory`, `fabric.extmemory`.

### Cycle Loop

```
for each cycle:
  driveBoundaryInputs()
  driveBoundaryOutputReady()
  // Phase 1
  repeat until stable:
    for module in combOrder_:
      module.evaluateCombinational()
  // Phase 2
  for module in seqModules_:
    module.advanceClock()
  advanceBoundaryState()
  commitErrors()
  emitTraceEvents()
  ++cycle
```

## Data Channel Path

### Input Channels

Host provides per-port input data as typed vectors:

```cpp
void setInput(unsigned portIdx, const std::vector<uint64_t> &data,
              const std::vector<uint16_t> &tags = {});
```

Input data is driven onto boundary input channels one element per cycle when
the channel is ready. Tagged inputs include per-element tag values.

### Output Channels

Host retrieves per-port output data after run completion:

```cpp
std::vector<uint64_t> getOutput(unsigned portIdx) const;
std::vector<uint16_t> getOutputTags(unsigned portIdx) const;
```

Output elements are collected from boundary output channels each cycle when
valid && ready.

### Completion Detection

The simulation run completes when:

- All input queues are fully consumed, AND
- All boundary output channels have been inactive for a configurable drain
  window, OR
- A maximum cycle count is reached (timeout)

## Trace and Performance Hooks

### Trace Events

The backend emits `LoomTraceEvent` records per
[spec-cosim-trace.md](./spec-cosim-trace.md). All 8 required event kinds are
supported:

- `EV_NODE_FIRE`: node produced output
- `EV_NODE_STALL_IN`: node blocked on input
- `EV_NODE_STALL_OUT`: node blocked on output ready
- `EV_ROUTE_USE`: edge/path segment active
- `EV_CONFIG_WRITE`: config word committed
- `EV_INVOCATION_START`: invocation entered running state
- `EV_INVOCATION_DONE`: invocation completed
- `EV_DEVICE_ERROR`: runtime error detected

### Performance Statistics

Per-node `PerfSnapshot` counters:

- `fireCycles`: cycles where node produced output
- `stallInCycles`: cycles blocked on input
- `stallOutCycles`: cycles blocked on output ready
- `idleCycles`: cycles with no activity

### Trace Overhead Controls

- Event-kind filtering: select which event kinds to collect
- Node/core filtering: select which hw_node_ids to trace
- Collection modes: Full (all events), Summary (counters only), Off

## Error Handling

### Runtime Errors

Fabric module runtime errors are detected per
[spec-fabric-error.md](./spec-fabric-error.md). Each module uses a two-phase
pending/commit error model:

1. `latchError(code)`: records pending error with min-code precedence
2. `commitError()`: promotes pending error to sticky error_valid/error_code

Runtime errors are reported as `EV_DEVICE_ERROR` trace events and included in
`SimResult`.

### Session Errors

- Invalid state transitions return descriptive error messages
- Configuration failures (missing config, size mismatch) are reported
- Simulation timeout is reported with cycle count reached
- Combinational loop detection is reported as a warning (simulation continues
  with fixed-point iteration)

## Output Artifacts

### Trace File

Written as `<dfg>_on_<adg>.trace` in the output directory. Contains serialized
`LoomTraceEvent` records suitable for viz playback.

### Statistics File

Written as `<dfg>_on_<adg>.stat` in the output directory. Contains JSON
performance statistics with per-node counters, derived metrics, and host timing.

## CPU Oracle Comparison

The backend supports output verification against CPU reference:

1. Host executes the same function with identical inputs on CPU
2. Accelerator outputs are compared element-by-element against CPU reference
3. Comparison result is PASS (all match) or FAIL (with mismatch details)
4. Partial comparison is marked invalid if run did not complete cleanly

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-architecture.md](./spec-cosim-architecture.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
- [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md)
- [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-error.md](./spec-fabric-error.md)

# FCC Simulation Specification

## Overview

FCC simulation is the execution-side validation layer for mapped designs.
The normative execution core is a shared cycle-accurate kernel that is reused
by:

- standalone mapped simulation
- runtime replay
- gem5 embedded execution

The simulated object is the mapped ADG hardware topology plus mapping overlay,
not the original software DFG alone.

## Input Sourcing

Standalone simulation must support synthetic or generated inputs, including:

- scalar and stream test vectors
- pre-filled external memory contents
- golden reference comparison inputs

The simulation interface must also support externally provided data when used by
host-side or gem5-driven flows.

## Session Contract

The simulation layer provides a session-like abstraction with operations for:

- build from mapped state
- config load
- input injection
- external-memory binding
- invoke or reset
- output retrieval
- trace and performance retrieval

This contract is shared conceptually with
[spec-host-accel-interface.md](./spec-host-accel-interface.md) and is realized
today by `SimSession`.

## Two-Phase Cycle Model

FCC's intended simulation model is cycle-accurate with two phases per cycle:

### Phase 1: Combinational Convergence

- drive boundary inputs
- evaluate combinational behavior of switches, PEs, FIFOs, and memory fronts
- propagate valid, ready, and data
- iterate until the combinational state converges or a limit is reached

Normative rule:

- `max_comb_iterations = 4`
- failure to converge within that bound is a structural simulation error

### Phase 2: Sequential Advance

- commit transfers
- update FIFO state
- update PE or temporal instruction state
- update memory completion state
- collect boundary outputs

The separation between combinational convergence and sequential state advance is
normative. Tokens produced during commit are visible starting in the next
cycle, not earlier in the same cycle.

## Invocation Boundary Semantics

The cycle kernel tracks:

- `done`
- `quiescent`
- `deadlock`

Normative rules:

- `quiescent` means no more transfer or fire can occur in the current mapped
  hardware state
- `done` means all software-visible completion obligations extracted from the
  mapping overlay are satisfied and the kernel is quiescent
- `deadlock` means the kernel is quiescent while completion obligations are
  still unsatisfied

Completion obligations come from the mapping overlay and include only
software-visible outputs and memory side effects. Residual non-obligation
control tails do not block completion.

## Module Families

The simulation architecture should accommodate at least these conceptual module
types:

- PE simulation
- switch simulation
- FIFO simulation
- memory and extmemory simulation
- stream and dataflow helper primitives

## Current Shared Kernel Surface

The shared cycle kernel currently exposes an interface equivalent in
responsibility to:

- `build(staticModel)`
- `configure(configImage)`
- `setInputTokens(portIdx, tokens)`
- `setMemoryRegionBacking(regionId, bytes, size)`
- `runUntilBoundary(maxCycles)`
- `getLastBoundaryReason()`
- `getOutputTokens(portIdx)`
- `getTraceDocument()`
- `getCurrentCycle()`

The current boundary reasons are:

- `NeedMemIssue`
- `WaitMemResp`
- `InvocationDone`
- `Deadlock`
- `BudgetHit`

## Channel Model

Simulation channels conceptually carry:

- valid
- ready
- data
- optional tag

The exact in-memory C++ representation may evolve, but these fields define the
architectural behavior.

## Trace and Statistics

FCC simulation should support:

- event traces
- per-node activity or stall accounting
- invocation start and completion markers
- a machine-readable result artifact containing resolved output tokens and
  post-execution memory-region snapshots

Current standalone artifact families include:

- `<mixed>.sim.setup.json`
- `<mixed>.sim.result.json`
- `<mixed>.sim.report.json`
- `<mixed>.sim.trace`
- `<mixed>.sim.stat`
- `<mixed>.simimage.json`
- `<mixed>.simimage.bin`

The trace artifact is a versioned JSON document. The top-level fields are:

- `version`
- `trace_kind`
- `producer`
- `epoch_id`
- `invocation_id`
- `core_id`
- `modules`
- `events`

This versioned trace is the contract consumed by HTML playback.

These outputs are relevant both for debugging and for later visualization or
performance analysis.

## Memory-Based Validation

Simulation must support validation by:

- output-port comparison
- post-execution memory comparison

This is required because many accelerator kernels communicate results primarily
through memory side effects.

## Runtime Image

FCC now emits a runtime image that captures the mapped static model and decoded
control bindings needed by the shared kernel. The runtime manifest records:

- `sim_image_json`
- `sim_image_bin`

The runtime image is the preferred handoff artifact for gem5 embedded
execution.

## Relationship to Other Specs

- [spec-host-accel-interface.md](./spec-host-accel-interface.md)
- [spec-dse.md](./spec-dse.md)
- [spec-fcc.md](./spec-fcc.md)

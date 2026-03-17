# FCC Simulation Specification

## Overview

FCC simulation is the execution-side validation layer for mapped designs.
It must support both standalone use and reuse inside a host-driven environment
such as gem5.

## Input Sourcing

Standalone simulation must support synthetic or generated inputs, including:

- scalar and stream test vectors
- pre-filled external memory contents
- golden reference comparison inputs

The simulation interface must also support externally provided data when used by
host-side or gem5-driven flows.

## Session Contract

The simulation layer is expected to support a session-like abstraction with
operations for:

- build from mapped state
- config load
- input injection
- external-memory binding
- invoke or reset
- output retrieval
- trace and performance retrieval

This contract is shared conceptually with
[spec-host-accel-interface.md](./spec-host-accel-interface.md).

## Two-Phase Cycle Model

FCC's intended simulation model is cycle-accurate with two phases per cycle:

### Phase 1: Combinational Convergence

- drive boundary inputs
- evaluate combinational behavior of switches, PEs, FIFOs, and memory fronts
- propagate valid, ready, and data
- iterate until the combinational state converges or a limit is reached

### Phase 2: Sequential Advance

- commit transfers
- update FIFO state
- update PE or temporal instruction state
- update memory completion state
- collect boundary outputs

The separation between combinational convergence and sequential state advance is
normative.

## Module Families

The simulation architecture should accommodate at least these conceptual module
types:

- PE simulation
- switch simulation
- FIFO simulation
- external-memory simulation
- stream and dataflow helper primitives

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

These outputs are relevant both for debugging and for later visualization or
performance analysis.

## Memory-Based Validation

Simulation must support validation by:

- output-port comparison
- post-execution memory comparison

This is required because many accelerator kernels communicate results primarily
through memory side effects.

## Relationship to Other Specs

- [spec-host-accel-interface.md](./spec-host-accel-interface.md)
- [spec-dse.md](./spec-dse.md)
- [spec-fcc.md](./spec-fcc.md)

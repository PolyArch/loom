# FCC Runtime and MMIO Specification

## Overview

FCC uses a runtime-facing session abstraction together with a simple MMIO
control plane for host-driven accelerator execution.

This document is the authority for the runtime lifecycle and MMIO contract.

## Runtime Responsibilities

A conforming FCC runtime must support:

- build from mapped DFG and ADG data
- configuration upload
- scalar or stream input setup
- external-memory region binding
- invocation and completion tracking
- output retrieval
- trace and performance retrieval

## SimSession Contract

The conceptual lifecycle API is centered on a session abstraction with methods
equivalent in responsibility to:

- `buildFromGraph(...)`
- `loadConfig(...)`
- `setInput(...)`
- `setExtMemoryBacking(...)`
- `invoke()`
- `resetExecution()`
- `resetAll()`
- `getOutput(...)`
- `getTrace()`
- `getPerfSnapshot(...)`

The exact surface syntax may change, but the lifecycle states are stable.

## Lifecycle Model

The session lifecycle is:

1. created
2. configured with mapped graph and config words
3. provisioned with inputs and memory regions
4. running
5. completed
6. optionally reset for reuse

Reconfiguration and execution must respect this order.

## Host Driver API

The host-side driver is expected to provide a thin API equivalent in behavior
to:

- `fcc_accel_init`
- `fcc_accel_load_config`
- `fcc_accel_set_mem_region`
- `fcc_accel_set_arg`
- `fcc_accel_launch`
- `fcc_accel_wait`
- `fcc_accel_cycle_count`

This API is intentionally minimal and suitable for baremetal execution.

## MMIO Register Model

The MMIO register file must cover at least these function groups:

- status
- control
- config address
- config data
- optional config size
- memory region base and size
- scalar arguments
- cycle count
- error code

## MMIO Semantics

The canonical host sequence is:

1. reset accelerator state
2. upload config words in ascending address order
3. bind memory regions
4. write scalar arguments
5. launch execution
6. poll or wait for completion
7. read cycle count and inspect outputs or memory side effects

Config upload is word-oriented and MMIO-based in the baseline FCC design.

## DMA vs MMIO Division

FCC uses MMIO for:

- control
- status
- configuration
- scalar setup

FCC uses DMA or equivalent backing access for:

- array and buffer payloads in external memory

Large memory payloads are not meant to be copied through MMIO data registers.

## Relationship to Backends

The same runtime lifecycle should work with:

- standalone simulation
- gem5-backed execution

The backend changes where MMIO requests terminate and how memory backing is
implemented, but the host-visible contract should remain the same.

## Related Documents

- [spec-host-accel-interface.md](./spec-host-accel-interface.md)
- [spec-gem5.md](./spec-gem5.md)
- [spec-simulation.md](./spec-simulation.md)

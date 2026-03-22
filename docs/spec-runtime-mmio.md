# LOOM Runtime and MMIO Specification

## Overview

LOOM uses a runtime-facing session abstraction together with a simple MMIO
control plane for host-driven accelerator execution.

This document is the authority for the runtime lifecycle and MMIO contract.

## Runtime Responsibilities

A conforming LOOM runtime must support:

- build from mapped DFG and ADG data
- configuration upload
- scalar or stream input setup
- external-memory region binding
- invocation and completion tracking
- output retrieval
- trace and performance retrieval

The runtime must be able to build execution either from mapping-time graph data
or from a prebuilt runtime image.

## SimSession Contract

The lifecycle API is centered on a session abstraction with methods equivalent
in responsibility to:

- `buildFromGraph(...)`
- `buildFromRuntimeImage(...)`
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

- `loom_accel_init`
- `loom_accel_load_config`
- `loom_accel_set_mem_region`
- `loom_accel_set_arg`
- `loom_accel_launch`
- `loom_accel_wait`
- `loom_accel_cycle_count`

This API is intentionally minimal and suitable for baremetal execution.

## Runtime Image and Control Image

The runtime layer now recognizes a runtime-image handoff family:

- `<mixed>.simimage.json`
- `<mixed>.simimage.bin`

The control image carried inside the runtime image includes at least:

- `start_token_port`
- scalar slot bindings
- memory-region slot bindings
- output slot bindings

This image is used by gem5 embedded execution and may also be used by runtime
replay.

## MMIO Register Model

The MMIO register file must cover at least these function groups:

- status
- control
- config blob base
- config blob size
- config load doorbell
- memory region base and size
- scalar arguments
- cycle count
- error code
- output selection and output token readback

## MMIO Semantics

The canonical host sequence is:

1. reset accelerator state
2. patch runtime-dependent fields inside the config image
3. provide config blob base and size
4. trigger config load
5. bind memory regions
6. write scalar arguments
7. launch execution
8. poll or wait for completion
9. read cycle count and inspect outputs or memory side effects

The host-visible config image remains a contiguous word stream, but the
preferred transport is bulk load of one contiguous blob rather than one-word
MMIO writes.

## DMA vs MMIO Division

LOOM uses MMIO for:

- control
- status
- configuration load setup
- scalar setup

LOOM uses DMA or equivalent backing access for:

- configuration blob transfer
- array and buffer payloads in external memory

Large memory payloads are not meant to be copied through MMIO data registers.

## Relationship to Backends

The same runtime lifecycle should work with:

- standalone simulation
- gem5-backed execution

The backend changes where MMIO requests terminate and how memory backing is
implemented, but the host-visible contract should remain the same.

In the current gem5 direct path, MMIO remains the control plane while memory
region payloads stay in gem5-visible memory.

`loom_accel_set_mem_region(slot, addr, size)` is the host-visible operation that
binds a software memory object to a runtime memory-region slot. The mapping-time
control image still defines which extmemory or memory region id uses which slot.

In the direct gem5 path, the concrete array or buffer address used by a mapped
extmemory or memory region comes from the uploaded config words:

- the host patches the relevant `addr_offset_table.base` entries inside the
  runtime config image before `loom_accel_load_config`
- the host then calls `loom_accel_set_mem_region(slot, addr, size)` to declare
  the legal host memory aperture and size that the gem5 device may access for
  that slot

This split lets the hardware-visible address source live inside the programmed
bitstream/config image while the runtime binding still provides a checked host
memory aperture for DMA and artifact export.

The device-side direct path is now:

- runtime image provides memory-region to slot bindings
- host patches `loom_runtime_config_words[]` before load
- host writes config blob base and size through MMIO
- host triggers one config-load operation
- the gem5 device DMA-loads the config blob into the accelerator config image
- the shared kernel emits `MemoryRequestRecord` batches at `NeedMemIssue`
- the gem5 device converts those records into DMA reads and writes
- DMA completions are pushed back as `MemoryCompletion` records

This means the primary embedded path no longer relies on invocation-wide memory
staging before or after execution.

## Related Documents

- [spec-host-accel-interface.md](./spec-host-accel-interface.md)
- [spec-gem5.md](./spec-gem5.md)
- [spec-simulation.md](./spec-simulation.md)

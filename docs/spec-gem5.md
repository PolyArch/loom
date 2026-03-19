# FCC gem5 Backend Specification

## Overview

FCC's gem5 integration provides a baremetal host-plus-accelerator execution
path in which the host CPU configures and launches the accelerator through MMIO
while bulk memory data is accessed through DMA.

## Architectural Boundary

The gem5 backend contains:

- a RISC-V baremetal host program
- a gem5 SimObject representing the accelerator device
- a shared simulation core reused from FCC standalone execution
- a DMA-backed memory adapter that connects extmemory activity to gem5 physical
  memory

## Baremetal Execution Model

The baseline FCC gem5 flow is baremetal:

- machine mode
- no OS
- no virtual-memory dependency
- physical addresses used directly for MMIO and DMA

This is a deliberate design choice to keep accelerator research isolated from
OS-stack complexity.

## Device Model

The accelerator device is expected to have a structure equivalent to a
`CgraDevice` SimObject that:

- registers an MMIO range
- owns or wraps the FCC simulation engine
- translates MMIO requests into runtime operations
- services DMA-like memory accesses through a memory-backing implementation

## Shared SimEngine Contract

The simulation engine used inside gem5 should be the same logical engine as the
standalone FCC simulation engine.

The critical abstraction is that the execution core does not need to know
whether it is running:

- in standalone mode with local memory buffers
- inside gem5 with DMA-backed physical memory

## Build Integration

The intended gem5 integration path uses gem5 `EXTRAS` support so the FCC device
can be built out-of-tree relative to the main gem5 source tree.

The build family also includes:

- the accelerator device sources
- gem5 Python configuration
- a baremetal host runtime and linker script

The current FCC build path uses:

- `module load scons`
- `scons -C externals/gem5 EXTRAS=<repo>/src/gem5dev build/RISCV/gem5.opt`

## File Family

The gem5-side integration is expected to include files equivalent in role to:

- `src/gem5dev/dev/fcc/FccCgraDevice.hh`
- `src/gem5dev/dev/fcc/FccCgraDevice.cc`
- `src/gem5dev/dev/fcc/FccCgraDevice.py`
- `SConscript`
- `runtime/baremetal/crt0.S`
- `runtime/baremetal/fcc_baremetal.ld`
- `tools/gem5/fcc_runtime_bridge.py`
- `tools/gem5/run_fcc_gem5_case.py`
- host driver implementation sources generated per case

The exact paths may evolve, but the role split is stable.

## Runtime Handoff Contract

The current FCC gem5 path reuses the standalone simulation core through a
runtime replay bridge instead of embedding a second accelerator simulator inside
the host program.

The handoff artifacts are:

- mapping-time runtime manifest:
  `out/e2e/<case>/<case>.runtime.json`
- gem5 device request payload:
  `invoke-*/request.json`
- replay result payload:
  `invoke-*/helper/runtime-result.json`
- exported trace and stat:
  `invoke-*/helper/runtime.trace`, `invoke-*/helper/runtime.stat`
- compact device reply metadata:
  `invoke-*/reply/reply.meta`

The runtime manifest must be sufficient to reconstruct the mapped execution
state needed by the simulator:

- DFG MLIR path
- ADG MLIR path
- config binary path
- software-node to hardware-node bindings
- scalar argument slot bindings
- memory-region slot bindings
- output slot bindings
- start-token binding

## End-to-End Execution Sequence

The intended gem5 execution flow is:

1. host program boots in baremetal mode
2. host resets the accelerator
3. host uploads config words by MMIO
4. host binds memory regions and scalar arguments by MMIO
5. host starts execution
6. gem5 device snapshots MMIO state and referenced physical-memory regions into
   a runtime request payload
7. gem5 device invokes FCC runtime replay through the bridge helper
8. FCC runtime replay executes the mapped accelerator with the shared simulator
   core and emits result, trace, and stat artifacts
9. gem5 device imports output tokens and updated memory back into gem5-visible
   state
10. device signals completion
11. host checks result data and exits with `m5_exit` on success or `m5_fail` on
    mismatch

## Current Smoke Flow

The repository-maintained smoke path is:

1. `./out/e2e/sum-array.sum-array-demo-chess-6x6/run.cmd`
2. `./out/e2e/sum-array.sum-array-demo-chess-6x6/run.gem5.cmd`

The case-local gem5 wrapper is expected to:

- rerun the normal FCC e2e flow through `run.cmd`
- invoke `tools/gem5/run_fcc_gem5_case.py`
- leave gem5 outputs under `out/e2e/<case>/gem5/`

The gem5 runner is expected to leave these per-case outputs:

- `gem5/host.c`
- `gem5/host.elf`
- `gem5/gem5.report.json`
- `gem5/<case>.gem5.trace`
- `gem5/<case>.gem5.stat`
- `gem5/accel-work/invoke-*/*`

## Trace and Performance

gem5-backed execution should expose the same logical trace and performance model
as standalone execution whenever practical.

If there are gem5-specific transport details, they must not change the event
semantics defined by FCC trace specs.

## Relationship to Other Specs

- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
- [spec-simulation.md](./spec-simulation.md)
- [spec-trace.md](./spec-trace.md)
- [spec-validation.md](./spec-validation.md)

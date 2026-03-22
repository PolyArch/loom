# LOOM gem5 Backend Specification

## Overview

LOOM's gem5 integration provides a baremetal host-plus-accelerator execution
path in which the host CPU configures and launches the accelerator through MMIO
while bulk memory data is accessed through DMA.

## Architectural Boundary

The gem5 backend contains:

- a RISC-V baremetal host program
- a gem5 SimObject representing the accelerator device
- a shared simulation core reused from LOOM standalone execution
- a DMA-backed memory adapter that connects extmemory activity to gem5 physical
  memory

## Baremetal Execution Model

The baseline LOOM gem5 flow is baremetal:

- machine mode
- no OS
- no virtual-memory dependency
- physical addresses used directly for MMIO and DMA

This is a deliberate design choice to keep accelerator research isolated from
OS-stack complexity.

## Device Model

The accelerator device is expected to have a structure equivalent to an
`LoomCgraDevice` SimObject that:

- registers an MMIO range
- owns or wraps the LOOM simulation engine
- translates MMIO requests into runtime operations
- services memory-backed region binding through a device-side adapter

## Shared SimEngine Contract

The simulation engine used inside gem5 should be the same logical engine as the
standalone LOOM simulation engine.

The critical abstraction is that the execution core does not need to know
whether it is running:

- in standalone mode with local memory buffers
- inside gem5 with DMA-backed physical memory

## Build Integration

The intended gem5 integration path uses gem5 `EXTRAS` support so the LOOM device
can be built out-of-tree relative to the main gem5 source tree.

The build family also includes:

- the accelerator device sources
- gem5 Python configuration
- a baremetal host runtime and linker script

The current LOOM build path uses:

- `module load scons`
- `scons -C externals/gem5 EXTRAS=<repo>/src/gem5dev build/RISCV/gem5.opt`

## File Family

The gem5-side integration is expected to include files equivalent in role to:

- `src/gem5dev/dev/loom/LoomCgraDevice.hh`
- `src/gem5dev/dev/loom/LoomCgraDevice.cc`
- `src/gem5dev/dev/loom/LoomCgraDevice.py`
- `SConscript`
- `runtime/baremetal/crt0.S`
- `runtime/baremetal/loom_baremetal.ld`
- `tools/gem5/run_loom_gem5_case.py`
- host driver implementation sources generated per case

The exact paths may evolve, but the role split is stable.

## Runtime Handoff Contract

The primary gem5 path now embeds the LOOM cycle kernel directly inside
`LoomCgraDevice`.

The primary handoff artifacts are:

- mapping-time runtime manifest:
  `out/e2e/<case>/<case>.runtime.json`
- mapping-time runtime image:
  `out/e2e/<case>/<case>.simimage.json`
- binary runtime image:
  `out/e2e/<case>/<case>.simimage.bin`

The runtime image must be sufficient to reconstruct the mapped execution state
needed by the device-local kernel:

- static mapped model
- config image
- start-token binding
- scalar argument slot bindings
- memory-region slot bindings
- output slot bindings

The primary repository-maintained gem5 path no longer depends on replay-bridge
CLI arguments or out-of-process bridge execution, and the legacy bridge helper
is no longer part of the repository-maintained execution path.

## End-to-End Execution Sequence

The current direct gem5 execution flow is:

1. host program boots in baremetal mode
2. host initializes software-visible input arrays and buffers in gem5-visible
   memory
3. host resets the accelerator
4. host patches the runtime config image so each extmemory or memory
   `addr_offset_table.base` entry points at the concrete host array or buffer
   address that this invocation should use
5. host writes config blob base and size by MMIO and triggers one config-load
   operation
6. gem5 device DMA-loads the patched config image from host memory into the
   accelerator config image
7. host binds memory-region slots and scalar arguments by MMIO
   The slot binding provides the legal host memory aperture and size for each
   mapped region domain; the extmemory base used by the hardware request path
   comes from the uploaded config image.
8. host starts execution
9. gem5 device rebuilds kernel state from the runtime image
10. gem5 device binds kernel-visible memory regions to gem5 DMA apertures and
   region sizes instead of staging entire memory images into local buffers
11. gem5 device advances the shared cycle kernel in-process with repeated
    `runUntilBoundary(maxCycles)` calls
12. `NeedMemIssue` causes the device to drain kernel-side outgoing memory
    requests and convert them into gem5 DMA transactions
13. DMA completions are pushed back into the shared kernel as
    `MemoryCompletion` records
14. `BudgetHit` causes the device to schedule another accelerator event after
    a batch-sized tick delay instead of blocking the current gem5 event until
    the whole invocation finishes
15. terminal boundary reasons such as `InvocationDone` or `Deadlock` end the
    invocation
16. gem5 device exports output tokens, trace, stat, and updated memory back
    into gem5-visible state
17. device signals completion
18. host reads back output ports or memory side effects, checks them against the
    host-side golden result, and exits with `m5_exit` on success or `m5_fail`
    on mismatch

The current repository configuration uses:

- `accel_cycle_latency = 1ns`
- `max_batch_cycles = 1024`
- `max_inflight_mem_reqs = 8`

for the embedded gem5 smoke path.

## Current Smoke Flow

The repository-maintained smoke path is:

1. `./out/e2e/sum-array.mesh-6x6-extmem-1/run.cmd`
2. `./out/e2e/sum-array.mesh-6x6-extmem-1/run.gem5.cmd`
3. `./out/e2e/vecadd.mesh-6x6-extmem-2/run.cmd`
4. `./out/e2e/vecadd.mesh-6x6-extmem-2/run.gem5.cmd`

The case-local gem5 wrapper is expected to:

- rerun the normal LOOM e2e flow through `run.cmd`
- invoke `tools/gem5/run_loom_gem5_case.py`
- leave gem5 outputs under `out/e2e/<case>/gem5/`

The primary invocation path only requires:

- `--accel-sim-image`
- `--accel-runtime-manifest`
- `--accel-work-dir`
- `--report`

The repository-maintained gem5 smoke coverage currently includes:

- `sum-array.mesh-6x6-extmem-1`
- `vecadd.mesh-6x6-extmem-2`
- `loom_gem5_sum-array_mesh-6x6-extmem-1_multiaccel`

The multi-device smoke target instantiates one primary accelerator plus one
additional idle LOOM accelerator in the same gem5 system and verifies that the
host-visible device still executes correctly.

The gem5 runner is expected to leave these per-case outputs:

- `gem5/host.c`
- `gem5/host.elf`
- `gem5/gem5.report.json`
- `gem5/<case>.cpu.trace`
- `gem5/<case>.cpu.stat.txt`
- `gem5/<case>.cpu.stat.json`
- `gem5/<case>.gem5.trace`
- `gem5/<case>.gem5.stat`
- `gem5/<case>.accel.stat.json`
- `gem5/<case>.gem5.memory.slot*.bin`
- `gem5/accel-work/invoke-*/*`

The direct path is still allowed to emit compatibility artifacts under
`accel-work`, but these are no longer evidence of replay-bridge execution by
themselves.

## DMA Spike

LOOM also maintains a minimal DMA integration spike that exercises:

- `DmaDevice`
- `getPort()` / `dma` wiring
- DMA read callback
- DMA write callback
- coexistence with MMIO control

This spike is intentionally isolated from `LoomCgraDevice` and exists to prove
the gem5 DMA path independently of the main accelerator device.

## Trace and Performance

gem5-backed execution should expose the same logical trace and performance model
as standalone execution whenever practical.

If there are gem5-specific transport details, they must not change the event
semantics defined by LOOM trace specs.

The repository-maintained gem5 flow now exports CPU-side and accelerator-side
artifacts separately:

- CPU-side trace and performance come from gem5 host execution:
  - `<case>.cpu.trace`
  - `<case>.cpu.stat.txt`
  - `<case>.cpu.stat.json`
- accelerator-side trace and performance come from the embedded LOOM cycle
  kernel:
  - `<case>.gem5.trace`
  - `<case>.gem5.stat`
  - `<case>.accel.stat.json`

## Relationship to Other Specs

- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
- [spec-simulation.md](./spec-simulation.md)
- [spec-trace.md](./spec-trace.md)
- [spec-validation.md](./spec-validation.md)

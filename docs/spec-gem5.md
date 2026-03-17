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

## File Family

The gem5-side integration is expected to include files equivalent in role to:

- `CgraDevice.hh`
- `CgraDevice.cc`
- `CgraDevice.py`
- `SConscript`
- `crt0.S`
- `fcc_baremetal.ld`
- `fcc_htif.c`
- host driver implementation sources

The exact paths may evolve, but the role split is stable.

## End-to-End Execution Sequence

The intended gem5 execution flow is:

1. host program boots in baremetal mode
2. host resets the accelerator
3. host uploads config words by MMIO
4. host binds memory regions and scalar arguments by MMIO
5. host starts execution
6. accelerator accesses array data through DMA
7. device signals completion
8. host checks result data and may print verdict output

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

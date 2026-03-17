# FCC Fabric Memory Interface Specification

## Overview

FCC memory-facing hardware uses explicit external-memory resources together with
switch routing, optional tag transforms, and a backend-independent memory
backing abstraction.

## MVP Rule

For the MVP flow, each memref-like software array is mapped to its own
extmemory-facing hardware resource.

This means:

- one logical array maps to one external-memory region
- vecadd-style kernels use separate regions for `a`, `b`, and `c`
- shared-memory aliasing and base-offset packing are not part of the MVP

## Single-Port vs Multi-Port Memory

When load and store counts are both one, a memory may be used without tagged
memory routing.

When `ldCount > 1` or `stCount > 1`, FCC uses tagged memory routing so multiple
logical streams can share one memory-facing endpoint.

## Tagged Multi-Port Memory Mechanism

For multi-port memory access:

1. each load or store stream is tagged with its logical port index
2. tagged streams are merged through routing fabric into the memory endpoint
3. the memory demultiplexes by tag internally
4. return data or completion tokens preserve tag identity on the way back

This mechanism requires:

- tag-add operations on ingress
- tagged switch routing across the relevant path
- tag-aware split or delete on egress

## Relationship to Switch Semantics

Tagged memory traffic still obeys the switch rules of the enclosing routing
resources. The only difference is that payload identity includes a tag field.

The memory tagging scheme does not authorize illegal structural merging in
untagged spatial-switch outputs.

## Memory Region Model

FCC treats external memory as a set of numbered regions. Each region has:

- a region id
- a base address
- a size
- a backing implementation

The accelerator runtime binds these regions before launch.

## Memory Backing Abstraction

The simulation core and gem5 device should share a backend-neutral memory
backing abstraction with equivalent responsibilities to:

- `read(regionId, byteOffset, numBytes)`
- `write(regionId, byteOffset, data, numBytes)`

Standalone mode may implement this over local arrays.
gem5 mode may implement this over DMA requests into simulated physical memory.

## Validation Implication

Memory side effects are architecturally observable outputs.
A run is not fully validated unless memory regions that are supposed to change
can be compared against their expected post-execution contents.

## Related Documents

- [spec-runtime-mmio.md](./spec-runtime-mmio.md)
- [spec-gem5.md](./spec-gem5.md)
- [spec-simulation.md](./spec-simulation.md)
- [spec-validation.md](./spec-validation.md)

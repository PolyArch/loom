# FCC Fabric Dialect Specification

## Overview

Fabric MLIR is FCC's hardware architecture IR. It describes the modules,
containers, routing resources, and memory endpoints that form the ADG.

FCC retains the overall Fabric role from Loom, but changes the internal PE and
switch model substantially.

## Key FCC Differences from Loom

- `fabric.spatial_pe` and `fabric.temporal_pe` contain explicit
  `fabric.function_unit` instances.
- `fabric.function_unit` may contain a configurable internal DAG via
  `fabric.static_mux`.
- `fabric.spatial_sw` may be decomposable, so routing may operate at sub-lane
  granularity.
- DFG-domain exploration replaces the older pragma-centric control model.

## Operation Families

| Family | Operations |
|--------|------------|
| Top level | `fabric.module`, `fabric.instance`, `fabric.yield` |
| Compute containers | `fabric.spatial_pe`, `fabric.temporal_pe` |
| Compute bodies | `fabric.function_unit`, `fabric.static_mux` |
| Routing | `fabric.spatial_sw`, `fabric.temporal_sw`, `fabric.fifo` |
| Memory | `fabric.memory`, `fabric.extmemory`, and related memory resources |

## Type Model

At module boundaries and inter-module connections, FCC uses structural bit
types, not native arithmetic types.

Typical rules:

- module and switch ports use `!fabric.bits<N>`
- tagged temporal routing uses `!fabric.tagged<!fabric.bits<N>, iK>`
- native types such as `i32`, `f32`, `index`, and `none` live inside
  `function_unit` boundaries

## Hardware Parameters vs Runtime Configuration

Each operation separates:

- hardware parameters: physical structure, fixed for an instance
- runtime configuration: values programmed by the mapper

For FCC, this split is especially important for:

- `spatial_sw` connectivity versus route tables
- `spatial_pe` structure versus opcode, mux, demux, and FU config selections
- `function_unit` static structure versus selected `static_mux` settings

When an ADG is given to the mapper, pre-populated runtime-config fields are
treated as hints unless a more specific spec says otherwise. The mapping output
is the authoritative source of final runtime configuration.

## Related Documents

- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal.md](./spec-fabric-temporal.md)

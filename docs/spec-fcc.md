# FCC Full-Pipeline Specification

## Overview

FCC is a fabric compiler that maps a software execution graph onto a hardware
architecture graph expressed in Fabric MLIR.

FCC inherits Loom's overall compiler shape, but introduces three major
architectural changes:

1. `function_unit` becomes an explicit level inside `spatial_pe` and
   `temporal_pe`.
2. A configurable hardware DAG is allowed inside `function_unit` through
   `fabric.mux`, making tech-mapping a hardware-software co-design step.
3. `spatial_sw` supports decomposable routing, so switch configuration may
   operate at sub-lane granularity instead of only logical-port granularity.

## Stage Structure

FCC uses the following stage boundaries:

1. Frontend and SCF-level analysis
2. DFG-domain exploration and candidate selection
3. DFG construction
4. ADG construction and flattening
5. Tech-mapping, placement, routing, and config generation
6. Visualization and optional simulation
7. Host or gem5 integration for end-to-end execution

The authoritative documents for these stage families are:

- CLI: [spec-cli.md](./spec-cli.md)
- Exploration: [spec-dse.md](./spec-dse.md)
- Fabric hardware: [spec-fabric.md](./spec-fabric.md)
- Mapper: [spec-mapper.md](./spec-mapper.md)
- Visualization: [spec-viz.md](./spec-viz.md)
- Host interface: [spec-host-accel-interface.md](./spec-host-accel-interface.md)

## High-Level Data Flow

The normative FCC flow is:

1. Lower input software into SCF-level MLIR.
2. Enumerate `dfg_domain` candidates and prune unfit candidates quickly.
3. Lower the selected candidate into DFG form (`handshake.func` plus FCC dataflow).
4. Build or load an ADG in Fabric MLIR.
5. Flatten the ADG into mapper-visible resources while preserving the
   information needed to reconstruct PE mux/demux routing and switch routes.
6. Run tech-mapping, placement, and routing.
7. Generate configuration artifacts, reports, and visualization.
8. Optionally run standalone simulation or host-driven execution.

## FCC-Specific Design Rules

- FCC does not use Loom's pragma system as the primary control surface.
  Acceleration domain selection is handled by exploration and cost modeling.
- `spatial_pe` ports and `function_unit` ports are not positional peers. The
  PE-level mux/demux fabric is the authority for how data reaches a selected FU.
- `spatial_sw` route selection is a mux model: each output chooses one input at
  the relevant granularity. Multiple inputs may not drive the same output in
  one configuration.
- `fabric.mux` configuration is selected during tech-mapping and is not
  rewritten during placement or routing.

## Scope of This Spec Set

This document is the top-level overview only. It does not redefine:

- Fabric operation semantics
- Mapper hard constraints
- Visualization interaction details
- Host execution protocol

Those are specified in the linked documents.

# LOOM vs the legacy design Specification Delta

## Overview

LOOM is based on the overall the legacy design architecture, but it is not a rename-only
fork. This document summarizes the normative architectural deltas that matter to
design, mapping, visualization, and future implementation work.

## What Stays Broadly Similar

LOOM keeps the same broad decomposition as the legacy design:

- software lowering to a DFG-like execution graph
- Fabric MLIR as the hardware architecture description
- mapper-driven place and route
- runtime configuration derived from mapping
- visualization and simulation as first-class outputs

This continuity is intentional. Existing the legacy design concepts remain useful reference
material unless an LOOM-specific spec overrides them.

## Main LOOM Deltas

### PE Structure

the legacy design's PE model is replaced by a containerized PE model:

- `spatial_pe` and `temporal_pe` contain explicit `function_unit` instances
- mapper-visible computation happens at FU granularity
- PE-local mux and demux routing becomes a first-class concern

### FU Internal Configurability

LOOM introduces configurable hardware DAGs inside `function_unit`:

- `fabric.mux` makes one physical FU represent multiple effective graphs
- tech-mapping must choose the FU configuration, not just the FU instance
- FU configuration is part of hardware-software co-design, not only backend
  encoding

### Switch Semantics

LOOM replaces the legacy design's `fabric.switch` naming with `fabric.spatial_sw` and adds:

- decomposable routing
- sub-lane route semantics
- fill and replication rules for wide outputs

### Frontend Control Surface

LOOM does not center the flow around the legacy design-style pragmas.
Instead:

- `dfg_domain` selection is an exploration dimension
- early design-space pruning is part of the intended pipeline

### Discard Model

LOOM removes the dependence on `handshake.sink` and uses a two-level discard
model:

- PE-local discard and disconnect
- switch-local discard bits

## Compatibility Boundary

the legacy design docs remain useful for:

- general compiler staging intuition
- broad mapper concepts
- visualization interaction patterns
- host/runtime organization

the legacy design docs are not authoritative for LOOM when they conflict with LOOM on:

- PE structure
- FU config model
- switch routing semantics
- visualization requirements for PE-internal and switch-internal mapped routes

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-mapper.md](./spec-mapper.md)
- [spec-viz.md](./spec-viz.md)

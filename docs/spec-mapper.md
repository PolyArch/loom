# Loom Mapper Specification

## Overview

The mapper performs place-and-route from a software execution graph to a
hardware resource graph.

- Software graph source: Handshake/Dataflow MLIR
- Hardware graph source: Fabric MLIR
- Result: a valid mapping plus runtime configuration values

This document defines mapper scope, interfaces, and stage boundaries. Detailed
content is intentionally split to avoid duplicated definitions:

- Data model and validity rules: [spec-mapper-model.md](./spec-mapper-model.md)
- Algorithm requirements and action semantics:
  [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)
- Cost/scoring model: [spec-mapper-cost.md](./spec-mapper-cost.md)

## Mapper Inputs and Outputs

### Inputs

- A software graph (`handshake.func`) that represents compute and
  communication semantics. The DFG must satisfy these preconditions:
  - No `handshake.fork` or `handshake.merge` operations (eliminated by
    the frontend during SCF-to-Handshake conversion).
  - SSA values may have multiple consumers (fan-out). The mapper handles
    fan-out through routing infrastructure (`fabric.switch` broadcast or
    `fabric.temporal_sw`).
- A hardware graph (`fabric.module`) that represents legal target resources
  and connectivity. The ADG is flattened (all `fabric.instance` inlined)
  before mapping begins.
- Optional mapping policy parameters (objective weights, search budget,
  deterministic seed).

### Outputs

- A `MappingState` that records placement and routing.
- Temporal assignments for temporal resources.
- Per-node configuration fragments merged into a full `config_mem` image.
- Diagnostics when no valid mapping exists under the selected policy.

The exact structure of `MappingState` and temporal metadata is defined only in
[spec-mapper-model.md](./spec-mapper-model.md).

## Functional Responsibilities

The mapper must perform all of the following:

1. Build candidate compatibility sets between software operations and hardware
   operations.
2. Assign software nodes to compatible hardware nodes.
3. Assign software edges to legal hardware paths.
4. Assign temporal resources (instruction slot, tag, opcode, register links)
   when temporal hardware is used.
5. Emit configuration values that conform to Fabric operation specifications.
6. Report failures with actionable diagnostics.

The mapper must not change hardware topology. It only selects legal usage of
existing hardware resources. Both the DFG and ADG are read-only inputs.

### Hardware Prerequisites

The mapper requires `addr_offset_table` support in `fabric.memory` and
`fabric.extmemory` for correct physical address translation and multi-region
memory mapping. See [spec-fabric-mem.md](./spec-fabric-mem.md) for the
`num_region` hardware parameter and `addr_offset_table` runtime
configuration.

## Relationship to ADG and Fabric Specs

- ADG defines hardware graph construction and export:
  [spec-adg.md](./spec-adg.md).
- Fabric defines operation semantics and constraints:
  [spec-fabric.md](./spec-fabric.md).
- `config_mem` layout rules are global and operation-specific:
  [spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

The mapper is valid only if its outputs satisfy those existing definitions.

## Mapping Validity

A mapping is valid when all hard constraints are satisfied, including:

- Type compatibility across mapped connections.
- Capacity/port constraints on hardware resources.
- Temporal constraints for tag width, slot count, and register legality.
- Route legality with respect to physical connectivity.
- Configuration encodings that satisfy all Fabric compile/config/runtime rules.

Normative hard constraints are defined in
[spec-mapper-model.md](./spec-mapper-model.md). Objective-driven preferences are
defined in [spec-mapper-cost.md](./spec-mapper-cost.md).

## Determinism and Reproducibility

Given identical inputs and deterministic policy settings, mapper output must be
reproducible. Sources of non-determinism (randomized search, tie breaks) must
be controlled by explicit seed/configuration parameters.

Algorithm-level deterministic tie-breaking and seed rules are defined in
[spec-mapper-algorithm.md](./spec-mapper-algorithm.md).

## Failure Modes

Mapper failure is expected when no feasible mapping exists under hard
constraints or resource limits.

Failure diagnostics should include:

- Unmapped software nodes/edges
- First hard-constraint violation class
- Minimal conflicting resource set when available

Diagnostic format is implementation-defined, but terms must reference concepts
from [spec-mapper-model.md](./spec-mapper-model.md).

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
- [spec-adg.md](./spec-adg.md)
- [spec-fabric.md](./spec-fabric.md)

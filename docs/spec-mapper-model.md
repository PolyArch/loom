# Loom Mapper Data Model Specification

## Overview

This document is the single source of truth for mapper data structures and
hard validity constraints.

It defines:

- Graph abstractions used by the mapper
- Mapping state representation
- Temporal assignment metadata
- Constraint classes required for a valid mapping
- Configuration emission model

Algorithm selection and search strategy are defined separately in
[spec-mapper-algorithm.md](./spec-mapper-algorithm.md).

## Graph Abstractions

### Software Graph

A directed graph where:

- Each node represents one software operation from Handshake/Dataflow MLIR.
- Each edge represents one software value/control dependency.
- Ports identify operand/result positions for each node.

### Hardware Graph

A directed graph derived from `fabric.module` where:

- Each node represents one hardware operation instance or module boundary port.
- Each edge represents one legal physical data path segment.
- Node and edge attributes carry capacities and type constraints from Fabric.

Fabric semantics remain authoritative in [spec-fabric.md](./spec-fabric.md).

## Canonical Mapping State

`MappingState` records both forward and reverse relations:

- `swNodeToHwNode`: software node -> hardware node assignment
- `swPortToHwPort`: software port -> hardware port assignment
- `swEdgeToHwPath`: software edge -> ordered hardware edge path
- `hwNodeToSwNodes`: hardware node -> assigned software node set
- `hwPortToSwPorts`: hardware port -> assigned software port set
- `hwEdgeToSwEdges`: hardware edge -> assigned software edge set

Implementations may choose any concrete container type, but semantics must
match this model.

### Path Representation

`swEdgeToHwPath` is an ordered list of connected hardware edge identifiers.
The first hardware edge must originate from the mapped source port, and the
last edge must terminate at the mapped destination port.

## Temporal Assignment Metadata

When a software node is mapped to temporal hardware (`fabric.temporal_pe`),
additional metadata is required:

- `slot`: instruction memory slot index
- `tag`: instruction match tag value
- `opcode`: FU selector index inside the temporal PE body

When an internal temporal register is used for routing:

- `register`: internal register index used by that mapped dependency

These assignments must satisfy all temporal constraints in
[spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md).

When a software edge is realized through `fabric.temporal_sw`, mapper metadata
must also capture:

- `temporal_sw_slot`: selected route-table slot index
- `temporal_sw_tag`: slot match tag value
- `temporal_sw_route`: enabled route mask for that slot

Encoding details for these fields are part of C6 and must follow
[spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md).

## Constraint Classes

A valid mapping must satisfy all classes below.

### C1: Node Compatibility

A software node can map to a hardware node only if operation semantics are
compatible (including body-op restrictions for `fabric.pe` and load/store
specialization rules). The authoritative `fabric.pe` body-op restrictions are
defined in [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md).

### C2: Port and Type Compatibility

Mapped ports must satisfy:

- Native/tagged category compatibility
- Value type compatibility
- Tag-width compatibility where tagged interfaces are used

No implicit conversion is allowed unless represented by explicit hardware
operations (`fabric.add_tag`, `fabric.del_tag`, `fabric.map_tag`, or cast PEs).

### C3: Route Legality

Each mapped software edge path must:

- Follow physically connected hardware edges
- Respect directionality
- Respect per-edge sharing or exclusivity constraints
- Connect the mapped source and destination ports

### C4: Capacity Constraints

Resources with bounded capacity (ports, buffers, route choices, temporal slots,
registers) must not exceed legal usage.

### C5: Temporal Constraints

For each temporal PE:

- `slot` is within `[0, num_instruction - 1]`
- `tag` fits interface tag bit width
- Duplicate tags inside configured instruction slots are not allowed
- Register indices are within `num_register`
- Register write-tag constraints follow Fabric temporal rules

### C6: Configuration Encoding Constraints

Generated configuration fields must conform to each operation's config format,
bit width, and error semantics:

- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)

This includes temporal-switch metadata (`temporal_sw_slot`,
`temporal_sw_tag`, `temporal_sw_route`) in addition to temporal-PE metadata.

## Resource Sharing Policy

This model distinguishes two classes:

- **Exclusive resources**: cannot be shared by multiple software assignments.
- **Shareable resources**: can represent multiple assignments if semantics allow.

The exact sharing policy is target-dependent and must be provided by the
hardware graph metadata. Algorithms must treat this policy as a hard constraint.

## Incremental Validity

Mapper implementations should support incremental checks:

- After node mapping update
- After edge routing update
- After temporal/config assignment update

Incremental checks must be equivalent to full-state validation.

## Configuration Emission Model

After mapping is valid, mapper emits per-node configuration fragments:

- `node_id`
- `config_bitstream` (packed, operation-local bit order)

A global assembler merges fragments into module-level `config_mem` words using
the authoritative allocation rules from
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

This two-level model avoids coupling mapping logic to physical address offsets.

## Related Documents

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)

# Loom Mapper Data Model Specification

## Overview

This document is the single source of truth for mapper data structures and
hard validity constraints.

It defines:

- Concrete graph data model used by the mapper
- Software graph (DFG) and hardware graph (ADG) properties
- Tech-mapping model for operation group matching
- Mapping state representation
- Temporal assignment metadata
- Constraint classes required for a valid mapping
- Configuration emission model

Algorithm selection and search strategy are defined separately in
[spec-mapper-algorithm.md](./spec-mapper-algorithm.md).

## Graph Data Model

The mapper uses a unified `Graph` container to represent both DFG and ADG.
All entities live in the `loom` namespace.

### Central Type Definitions

```cpp
using IdIndex = uint32_t;
constexpr IdIndex INVALID_ID = static_cast<IdIndex>(-1);
```

`IdIndex` is the single index/ID type used throughout the mapper. The invalid
sentinel is all-ones (0xFFFFFFFF). This type is centralized so it can be
widened to `uint64_t` if needed in the future.

### ID-as-Index Principle

No separately allocated IDs exist. The position of an entity in its owning
vector IS its ID. This gives O(1) lookup with zero hash overhead.

Deletion sets the vector slot to `nullptr` without shifting subsequent
entries, so existing IDs remain stable. This means vectors may contain null
gaps after deletions.

### Port

```cpp
class Port {
public:
  IdIndex parentNode = INVALID_ID;
  llvm::SmallVector<IdIndex, 2> connectedEdges;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};
```

A port is the basic connector. It records:

- `parentNode`: ID of the owning node in `Graph::nodes`.
- `connectedEdges`: IDs of edges in `Graph::edges` connected to this port.
  For DFG output ports, multiple edges represent SSA value fan-out (one edge
  per consumer). For ADG ports, this list has at most one entry because
  implicit fan-out is not allowed at the hardware level
  (data duplication requires explicit `fabric.switch` broadcast).
- `attributes`: MLIR-native metadata (value type, tag width, direction, port
  role for memory ports, etc.).

### Edge

```cpp
class Edge {
public:
  IdIndex srcPort = INVALID_ID;
  IdIndex dstPort = INVALID_ID;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};
```

An edge connects exactly one source port to one destination port. In DFG it
represents a data or control dependency. In ADG it represents a physical
connection segment.

### Node

```cpp
class Node {
public:
  enum class Kind {
    OperationNode,     // Software operation (DFG) or hardware resource (ADG)
    ModuleInputNode,   // Sentinel: represents handshake.func block argument
    ModuleOutputNode,  // Sentinel: represents handshake.return operand
  };

  Kind kind = Kind::OperationNode;
  llvm::SmallVector<IdIndex, 4> inputPorts;
  llvm::SmallVector<IdIndex, 4> outputPorts;
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
};
```

A node is a collection of ports. Port lists store global port IDs
(`Graph::ports` indices). Removing a port from a node compacts the list (no
holes in port lists). The invariant `inputPorts.size() + outputPorts.size()
>= 1` must hold (no portless nodes).

In DFG, a node represents a software operation, a module input sentinel, or
a module output sentinel. In ADG, a node represents a hardware resource.

**Sentinel nodes** (both DFG and ADG):

- `ModuleInputNode`: In DFG, represents a `handshake.func` block argument;
  in ADG, represents a `fabric.module` input argument. Has one output port
  connected to internal consumers via edges.
- `ModuleOutputNode`: In DFG, represents a `handshake.return` operand; in
  ADG, represents a `fabric.module` output. Has one input port connected
  from an internal producer.
- Sentinel nodes are never placement targets (never appear in
  `swNodeToHwNode`). They are fixed at the module I/O boundary.
- The mapper binds DFG sentinel ports to ADG sentinel ports via `MapPort`,
  subject to type and bit-width compatibility (C2).
- Edges incident to sentinel nodes participate in routing like any other
  edge. They are routed from the bound hardware sentinel port through the
  ADG routing network to/from the placed operation's hardware port.

### Graph Container

```cpp
class Graph {
public:
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::unique_ptr<Port>> ports;
  std::vector<std::unique_ptr<Edge>> edges;

  IdIndex addNode(std::unique_ptr<Node> node);
  IdIndex addPort(std::unique_ptr<Port> port);
  IdIndex addEdge(std::unique_ptr<Edge> edge);

  void removeNode(IdIndex id);
  void removePort(IdIndex id);
  void removeEdge(IdIndex id);

  Node* getNode(IdIndex id) const;
  Port* getPort(IdIndex id) const;
  Edge* getEdge(IdIndex id) const;

  bool isValid(IdIndex id, EntityKind kind) const;
  size_t countNodes() const;  // non-null entries
  size_t countPorts() const;
  size_t countEdges() const;

  mlir::MLIRContext* context = nullptr;
};
```

Design properties:

1. **ID stability**: deletion marks slots as `nullptr`, never shifts.
2. **Pre-allocation**: vectors are reserved to expected max capacity on
   construction to avoid reallocation.
3. **Bidirectional references**: Port stores parent node; Node stores port
   list. Edge stores two port IDs; Port stores connected edge IDs.
   Intentional redundancy for O(1) neighbor traversal.
4. **MLIR attributes**: `mlir::NamedAttribute` vectors provide extensibility
   and direct interop with the dialect system.

**Deletion cascade rules (ownership-based):**

- Remove Node: for each port in the node's port lists, if
  `port.parentNode == this node's ID` (this node owns the port), cascade
  `removePort`. Otherwise, only remove the port reference from this node's
  port list without deleting the port itself. This distinction is critical
  for temporal PE FU nodes that reference ports owned by the virtual node.
- Remove Edge: remove the edge ID from both endpoint ports'
  `connectedEdges`.
- Remove Port: remove from parent node's port list; remove all connected
  edges.

## Software Graph (DFG)

The DFG is extracted from `handshake.func` within Handshake+Dataflow MLIR.

### Extraction Rules

1. Each MLIR operation inside `handshake.func` becomes a Node.
2. Each SSA value use becomes an Edge from the producing operation's output
   Port to the consuming operation's input Port.
3. Each MLIR result type becomes Port attributes (value type, tag metadata).
4. Block arguments (function parameters) become `ModuleInputNode` sentinels,
   each with one output port.
5. `handshake.return` operands become `ModuleOutputNode` sentinels, each
   with one input port.

### DFG Properties

- **No fork/merge**: `handshake.fork` and `handshake.merge` are absent from
  the DFG. The frontend eliminates them during SCF-to-Handshake conversion.
  `handshake.mux` replaces merge semantics; `handshake.cond_br` and dataflow
  operations replace fork semantics.
- **SSA fan-out**: A single SSA result may have multiple consumers. This is
  represented as one output Port with multiple connected Edges. During
  mapping, fan-out is realized through `fabric.switch` broadcast or
  `fabric.temporal_sw`. The mapper must route from one hardware output port
  to multiple hardware input ports through routing infrastructure.
- **Control tokens**: `none`-typed edges (control tokens) are first-class
  edges with the same treatment as data edges.
- **Memory operations**: `handshake.memory` and `handshake.extmemory`
  produce multiple results (load data, load done, store done). Each result
  becomes a separate output port.

### DFG Module I/O Sentinels

Block arguments and return operands are represented as sentinel nodes:

1. For each block argument of `handshake.func`, create a `ModuleInputNode`
   with one output port. Edges connect this output port to all operations
   that consume the argument.
2. For each operand of `handshake.return`, create a `ModuleOutputNode`
   with one input port. An edge connects from the producing operation's
   output port to this input port.

### ADG Module I/O Sentinels

`fabric.module` inputs and outputs are represented as sentinel nodes in
the flattened ADG:

1. For each `fabric.module` input argument, create a `ModuleInputNode`
   with one output port. An edge connects this output port to the internal
   component that receives this input (per the 1-to-1 module connectivity
   rule).
2. For each `fabric.module` output, create a `ModuleOutputNode` with one
   input port. An edge connects from the internal component that produces
   this output to this input port.

Port types on ADG sentinel nodes follow `fabric.module` port ordering:
`memref*`, `native*`, `tagged*`.

### Sentinel Mapping Semantics

Sentinel nodes:
- Are not placement targets (never appear in `swNodeToHwNode`).
- Have ports that are bound between DFG and ADG via `MapPort`, subject
  to type and bit-width compatibility (C2).
- Edges incident to sentinel nodes are routed through the ADG connectivity
  network, just like edges between operation nodes. Routing paths start
  at the bound hardware sentinel port and end at the placed operation's
  hardware port (or vice versa).
- Input sentinels must have at least one outgoing edge (all inputs are used).
- Output sentinels must have exactly one incoming edge (each return value
  has a single producer).
- Fan-out from input sentinels (a block argument consumed by multiple
  operations) is handled through `fabric.switch` broadcast, following
  the same fan-out routing rules as operation output ports.

### DFG Immutability

The mapper must not modify the DFG. It is a read-only input.

## Hardware Graph (ADG)

The ADG is derived from `fabric.module` in Fabric MLIR.

### ADG Flattening

Before graph extraction, the ADG undergoes flattening: all
`fabric.instance` references are inlined, removing hierarchy. The result is
a two-level structure:

```
fabric.module
  +-- functional units:  fabric.pe, fabric.temporal_pe
  +-- routing network:   fabric.switch, fabric.temporal_sw,
                          fabric.add_tag, fabric.map_tag, fabric.del_tag,
                          fabric.fifo
  +-- memory:            fabric.memory, fabric.extmemory
```

Exception: `fabric.temporal_pe` contains internal `fabric.pe` functional
units, creating at most three-level nesting
(`fabric.module > fabric.temporal_pe > fabric.pe`). However, a temporal PE
is logically a group of constrained PEs (see below).

### ADG Resource Classes

After flattening, every hardware node belongs to one of four classes:

| Class | Node Types | Mapper Role |
|-------|-----------|-------------|
| Functional | `fabric.pe`, `fabric.temporal_pe` (virtual) | Placement targets |
| Routing | `fabric.switch`, `fabric.temporal_sw`, `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`, `fabric.fifo` | Routing path elements |
| Memory | `fabric.memory`, `fabric.extmemory` | Placement targets |
| Boundary | `ModuleInputNode`, `ModuleOutputNode` | Sentinel port binding targets |

Software operations map only to functional and memory nodes.
Routing nodes serve software edges, not software operations.
Boundary sentinel nodes serve module I/O port binding, not operation
placement.

### ADG Immutability

The mapper must not modify the ADG. It is a read-only input.

### Graph Immutability During Mapping

Both DFG and ADG are frozen (read-only) during the entire PnR process.
The mapper operates exclusively on `MappingState`, which tracks
software-to-hardware assignments without mutating graph structure.

The ownership-based deletion API exists in the `Graph` class for
potential future use cases (e.g., hardware design space exploration)
but is never invoked during mapper execution.

### Temporal PE Representation

A `fabric.temporal_pe` with N internal functional units (FUs) creates N+1
nodes in the ADG graph:

- N nodes representing the internal `fabric.pe` FUs. These are the actual
  placement targets for software operations.
- 1 virtual node representing the `fabric.temporal_pe` container. This node
  provides constraint context but is never a placement target.

All N+1 nodes share the same set of ports. The ports' `parentNode` field
points to the virtual temporal PE node (the physical owner). Each FU node's
`inputPorts` and `outputPorts` lists reference the same global port IDs.

FU node attributes record:

- The temporal PE virtual node ID they belong to.

Virtual node attributes record:

- IDs of all contained FU nodes.
- `num_instruction`: maximum total software operations across all FUs.
- `num_register`: maximum internal register-routed edges.
- Other temporal PE hardware parameters.

The mapping target for `swNodeToHwNode` is always an FU node ID, never
the virtual node ID. The virtual node exists to enforce coupled constraints
across FUs (instruction count, register count).

### Connectivity Matrix

To support routing, the mapper builds a global connectivity matrix from
the ADG. This matrix records which output ports can reach which input ports
through routing nodes.

Construction:

- For each ADG edge (physical connection): `outputPortId -> inputPortId`.
- For each routing node: `inputPortId -> outputPortId(s)` based on internal
  routing semantics:
  - `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`, `fabric.fifo`:
    single input to single output (pass-through).
  - `fabric.switch`: input to outputs per `connectivity_table`.
  - `fabric.temporal_sw`: input to outputs per `connectivity_table`.

A routing query asks: given a source hardware output port and a destination
hardware input port, find a legal path through the connectivity matrix.

### Port Type Classification

ADG ports belong to one of these categories:

- **Native**: untagged value type (e.g., `i32`, `f32`).
- **Tagged**: `!dataflow.tagged<value_type, tag_type>`.
- **Memory**: specialized port roles (`addr`, `data`, `ctrl`, `done`).

Port categories constrain which DFG ports can map to which ADG ports (C2).

## Tech-Mapping Model

Tech-mapping determines which groups of DFG operations can map to which
hardware PE bodies. This applies to `fabric.pe` nodes whose bodies contain
one or more operations.

### Operation Groups

A `fabric.pe` body defines an operation group: the set of MLIR operations
between the entry block and `fabric.yield`. An operation group may contain:

- A single operation (most common): e.g., one `arith.addi`.
- Multiple connected operations (a small subgraph): e.g.,
  `arith.addi` feeding `arith.muli`.

An operation group is the atomic unit of placement. It maps as a whole or
not at all; partial use of a multi-operation PE body is not allowed.

### Subgraph Isomorphism

For multi-operation PE bodies, the mapper must find subgraph isomorphism
matches in the DFG. Given a PE body subgraph P and the DFG G, a match is a
mapping from P's operations to G's operations that preserves:

- Operation names and semantics.
- Data flow connectivity (producer-consumer edges).
- Operand/result position correspondence.

Since PE bodies are typically small (under 10 operations), exhaustive
matching is tractable. The matcher should:

1. Enumerate unique PE body patterns (deduplicate identical bodies).
2. For each pattern, scan the DFG for all valid matches.
3. Record each match as a candidate operation group.

### Runtime-Configurable Parameters

Certain operation attributes are runtime-configurable and do not affect
compatibility. The following are NOT checked during tech-mapping:

| Operation | Runtime-configurable parameter |
|-----------|-------------------------------|
| `arith.cmpi` | predicate (4-bit config in `config_mem`) |
| `arith.cmpf` | predicate (4-bit config in `config_mem`) |
| `dataflow.stream` | `cont_cond` (5-bit one-hot config in `config_mem`) |

The `step_op` of `dataflow.stream` IS a hardware parameter and must match.

### Candidate Set Construction

`CandidateSet(swNode)` is the set of ADG nodes where a software node (or
software operation group) can legally be placed.

For single-operation nodes, the candidate set includes all `fabric.pe`
nodes whose body contains a compatible single operation, plus temporal PE
FU nodes whose body matches.

For operation groups (matched via subgraph isomorphism), the candidate set
includes `fabric.pe` nodes whose body subgraph matches the group.

Memory nodes have their own rules (see C4 capacity constraints).

If any software operation has an empty candidate set, the mapper reports
early failure (`CPL_MAPPER_NO_COMPATIBLE_HW`).

### Exclusivity Rules

The following `fabric.pe` body exclusivity rules from
[spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md) constrain tech-mapping:

- **Constant Exclusivity**: `handshake.constant` must be the sole operation.
- **Load/Store Exclusivity**: `handshake.load` or `handshake.store` must be
  the sole operation.
- **Dataflow Exclusivity**: `dataflow.{carry,invariant,stream,gate}` must
  each be the sole operation.
- **Homogeneous Consumption**: Full-consume/produce and partial-consume/produce
  operations cannot mix in the same body.

## Canonical Mapping State

`MappingState` records both forward and reverse relations:

- `swNodeToHwNode`: software node -> hardware node assignment.
  For operation groups, multiple software node IDs map to the same hardware
  node ID.
- `swPortToHwPort`: software port -> hardware port assignment.
- `swEdgeToHwPaths`: software edge -> ordered hardware port-pair path.
  For fan-out edges, each consumer edge is routed independently.
- `hwNodeToSwNodes`: hardware node -> assigned software node set.
  Multiple entries indicate either operation-group mapping or temporal
  sharing.
- `hwPortToSwPorts`: hardware port -> assigned software port set.
- `hwEdgeToSwEdges`: hardware edge -> assigned software edge set.
  Multiple entries indicate tagged edge sharing.

The essential mapping is `swNodeToHwNode`. All other mappings are derivable
but stored explicitly for performance.

### Path Representation

`swEdgeToHwPaths` is an ordered list of port-pair identifiers. Each
software edge maps to a sequence of hardware edges (port-pairs) forming
a multi-hop route through the ADG routing network. The first entry
originates from the mapped source port, and the last terminates at the
mapped destination port. Intermediate entries traverse routing nodes
(switches, temporal switches, FIFOs, tag operations).

The plural naming reflects that one software edge corresponds to
potentially many hardware edges (a path through multiple routing hops).

### Action Cascade Semantics

The action primitives have the following cascade behavior on undo:

- `UnmapNode(swNode)`: unmaps the node assignment; automatically
  unmaps all ports of this node via `UnmapPort`; each port unmap
  cascades to unmapping all edges connected through that port.
- `UnmapPort(swPort)`: unmaps the port binding; cascades to
  `UnmapEdge` for all software edges connected to this port (since
  the route endpoints become invalid).
- `UnmapEdge(swEdge)`: unmaps the edge route; removes all hardware
  edge/port entries for this route from `swEdgeToHwPaths` and from
  the reverse mapping `hwEdgeToSwEdges`.

Forward actions do NOT cascade:

- `MapNode(swNode, hwNode)` may trigger default port alignment
  (side effect), but does not automatically route edges.
- `MapPort(swPort, hwPort)` binds a single port pair. It does not
  route connected edges.
- `MapEdge(swEdge, pathHint)` routes a single edge. It does not
  affect node or port mappings.

This asymmetry (cascade on undo, no cascade on do) ensures that
undoing a high-level assignment consistently cleans up all dependent
state, while constructing a mapping requires explicit step-by-step
commitment.

### Tagged Edge Sharing

Multiple software edges may share a single hardware edge if:

- The hardware edge carries a tagged type.
- Each software edge is assigned a distinct tag value.
- The number of shared edges does not exceed `2^TAG_WIDTH`.

Tag values are assigned by the mapper during routing. They must be
allocated to avoid conflicts along the entire shared path.

## Temporal Assignment Metadata

When a software node is mapped to a temporal PE FU, additional metadata:

- `slot`: instruction memory slot index.
- `tag`: instruction match tag value.
- `opcode`: FU selector index inside the temporal PE body.

Slot assignment is trivial (sequential from 0) because all slots are
equivalent. The constraint is that total mapped operations across all FUs
in one temporal PE must not exceed `num_instruction`.

When an internal temporal register is used:

- `register`: internal register index for a mapped dependency edge
  between operations sharing the same temporal PE.

Register assignment is also position-independent (all registers are
equivalent). The constraint is that total register-routed edges must not
exceed `num_register`.

When a software edge passes through `fabric.temporal_sw`:

- `temporal_sw_slot`: selected route-table slot index.
- `temporal_sw_tag`: slot match tag value.
- `temporal_sw_route`: enabled route mask for that slot.

## Constraint Classes

A valid mapping must satisfy all classes below.

### C1: Node Compatibility

A software node can map to a hardware node only if operation semantics are
compatible. Compatibility is determined by:

1. The hardware node's body operations matching the software operation(s)
   per [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md).
2. Tech-mapping rules (operation group matching, exclusivity).
3. Load/store PE specialization (TagOverwrite vs TagTransparent) per
   [spec-fabric-pe.md](./spec-fabric-pe.md).
4. Memory type matching (`handshake.memory` -> `fabric.memory`,
   `handshake.extmemory` -> `fabric.extmemory`).

### C2: Port and Type Compatibility

Mapped ports must satisfy:

- Native/tagged category compatibility (native cannot connect to tagged
  without explicit `fabric.add_tag`, `fabric.del_tag`, or `fabric.map_tag`).
- Tag-width compatibility where tagged interfaces are used.
- **Routing node type relaxation**: for the six routing node types
  (`fabric.switch`, `fabric.temporal_sw`, `fabric.add_tag`,
  `fabric.map_tag`, `fabric.del_tag`, `fabric.fifo`), type compatibility
  checks only bit width, not semantic type. Specifically:
  - Native-to-native: total bit width must match (e.g., `i32` and `f32`
    are compatible through a switch because both are 32 bits).
  - Tagged-to-tagged: value bit width and tag bit width must each match
    (value semantic types may differ).
  - Native-to-tagged: never allowed, even through routing nodes.

### C3: Route Legality

Each mapped software edge path must:

- **(C3.1)** Follow physically connected hardware edges.
- **(C3.2)** Respect directionality.
- **(C3.3)** Respect per-edge sharing or exclusivity constraints.
- **(C3.4)** Connect the mapped source and destination hardware ports. For edges
  incident to sentinel nodes, the sentinel endpoint is the bound hardware
  sentinel port (via `MapPort`); for edges between operation nodes, both
  endpoints are placed operation ports.
- **(C3.5)** Preserve memory done-token wiring legality.
- **(C3.6)** For fan-out edges: routing through `fabric.switch` broadcast or
  `fabric.temporal_sw` is mandatory; implicit hardware fan-out is illegal.
  This applies equally to fan-out from sentinel input nodes and from
  operation output ports.

### C4: Capacity Constraints

Resources with bounded capacity must not exceed legal usage.

**Edge capacity**:

- **(C4.1)** A native (untagged) hardware edge may carry at most one
  software edge (exclusive use).
- **(C4.2)** A tagged hardware edge may carry at most `2^TAG_WIDTH`
  software edges, each with a distinct tag value.

**Memory capacity** (extended with `addr_offset_table`):

- **(C4.3)** `handshake.memory` maps to `fabric.memory`;
  `handshake.extmemory` maps to `fabric.extmemory`.
  `memref` element type must match.
  For `fabric.memory`: total capacity of mapped software memories must not
  exceed hardware memory capacity.
- **(C4.4)** Sum of mapped `ldCount` values must not exceed hardware `ldCount`.
  Sum of mapped `stCount` values must not exceed hardware `stCount`.
- **(C4.5)** Number of distinct mapped software memory operations must not exceed
  `num_region` (the `addr_offset_table` size).
- **(C4.6)** `addr_offset_table` entries assign base addresses and tag ranges for
  each mapped software memory region. See
  [spec-fabric-mem.md](./spec-fabric-mem.md).

**Other capacity authorities** (unchanged):

- Switch: [spec-fabric-switch.md](./spec-fabric-switch.md)
- Temporal switch: [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- Temporal PE: [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- PE: [spec-fabric-pe.md](./spec-fabric-pe.md)

### C5: Temporal Constraints

For each temporal PE:

- **(C5.1)** Total mapped operations across all FUs must not exceed `num_instruction`.
- **(C5.2)** Total register-routed edges must not exceed `num_register`.
- **(C5.3)** `tag` values fit interface tag bit width.
- **(C5.4)** No duplicate tags within configured instruction slots.

### C6: Configuration Encoding Constraints

Generated configuration fields must conform to each operation's config
format, bit width, and error semantics. Authorities:

- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-mem.md](./spec-fabric-mem.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)

Derived configuration values:

- `constant_value`: from mapped software constant payload.
- `cont_cond_sel`: from mapped software stream-continue predicate.
- `output_tag`: from mapper tag-assignment policy for TagOverwrite PEs.
- `addr_offset_table` entries: from mapper memory region assignment.
- Compare predicates: from mapped software compare operation predicate.

## Resource Sharing Policy

Two classes:

- **Exclusive resources**: cannot be shared (e.g., native hardware edges).
- **Shareable resources**: can carry multiple assignments (e.g., tagged
  hardware edges with distinct tag values, temporal PE instruction slots).

Sharing policy is determined by hardware graph metadata and is a hard
constraint.

## Incremental Validity

Mapper implementations should support incremental checks after each action
(node mapping, edge routing, temporal assignment). Incremental checks must
be equivalent to full-state validation.

## Configuration Emission Model

After mapping is valid, the mapper emits per-node configuration fragments:

- `node_id`
- `config_bitstream` (packed, operation-local bit order)

A global assembler merges fragments into module-level `config_mem` words
using allocation rules from
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Related Documents

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-mem.md](./spec-fabric-mem.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)

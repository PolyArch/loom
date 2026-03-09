# System Dialect Specification

## Overview

The `system` dialect defines the MLIR operations for describing a
heterogeneous multi-core CGRA system. It sits above the `fabric` dialect in
the Loom IR hierarchy and provides hardware topology, inter-core communication,
and application-level kernel graph representations.

The system dialect contains two categories of operations:

- **Hardware topology**: `system.design`, `system.core`, `system.link`,
  `system.router`, `system.cache`, `system.extmem` -- describe multi-core
  physical structure
- **Application model**: `system.kernel_dag` and its nested kernel/dependency
  constructs -- describe multi-kernel software workloads

Hardware topology ops reference `fabric.module` definitions but do not modify
them. Application model ops reference `handshake.func` definitions.

## Operation Summary

| Category | Operation | Purpose |
|----------|-----------|---------|
| Container | `system.design` | Top-level multi-core system |
| Hardware | `system.core` | Single CGRA core wrapping a `fabric.module` |
| Hardware | `system.link` | Point-to-point inter-core connection segment |
| Hardware | `system.router` | Intermediate NoC routing node |
| Hardware | `system.cache` | Analytical cache node in the memory hierarchy |
| Hardware | `system.extmem` | External memory interface endpoint |
| Application | `system.kernel_dag` | Multi-kernel application dependency graph |

## Operation: `system.design`

A `system.design` is the top-level container for a heterogeneous multi-core
system. It holds all cores, NoC infrastructure, cache nodes, and the
application kernel graph.

### Syntax

```mlir
system.design @name {
  // system.core, system.link, system.router, system.cache, system.kernel_dag
}
```

### Body

The body uses Graph region semantics (forward references allowed). Only
system dialect operations are allowed at the top level.

Allowed operations:
- `system.core`
- `system.link`
- `system.router`
- `system.cache`
- `system.extmem`
- `system.kernel_dag`

### Constraints

- Must contain at least one `system.core`.
- Must contain at most one `system.kernel_dag` (may be absent if the design
  is hardware-only).
- All `system.link` endpoints must reference cores or routers defined
  within the same `system.design`.

## Operation: `system.core`

A `system.core` represents a single CGRA accelerator core within a
multi-core system.

### Syntax

```mlir
system.core @name {
  module_ref = @fabric_module_symbol,
  frequency = 1.0 : f64
}
```

### Attributes

- `module_ref` (symbol reference): references a `fabric.module` definition
  that describes this core's hardware. The referenced module must exist.
- `frequency` (f64, optional): clock frequency in GHz. Different cores
  may have different frequencies for analytical bandwidth normalization.
  The analytical model uses per-core frequency to compute absolute
  throughput (tokens/second = tokens/cycle * frequency). This does not
  imply multi-clock-domain hardware generation; see non-goals in
  [spec-hetero.md](./spec-hetero.md). Default: 1.0.

### Capability Summary

The system framework extracts a **capability summary** from each core's
referenced `fabric.module`. This summary is a derived, read-only view:

```
CoreCapability {
  supported_ops : set<string>     // union of all PE body op names
  pe_count      : int             // total fabric.pe + temporal PE FU count
  memory_capacity : int           // total on-chip scratchpad bytes
  memory_ports  : int             // total load + store port count
  temporal_slots : int            // max instructions per temporal PE (0 if none)
  frequency     : f64             // from system.core attribute
}
```

Extraction rules:

- `supported_ops`: walk all `fabric.pe` and temporal PE FU bodies in the
  referenced `fabric.module`, collect the set of MLIR operation names
  (e.g., "arith.addi", "arith.mulf", "handshake.constant").
- `pe_count`: count all `fabric.pe` instances plus all temporal PE FU
  instances (not virtual temporal PE nodes).
- `memory_capacity`: sum of all `fabric.memory` memref byte sizes.
  `fabric.extmemory` does not contribute (external memory is unbounded).
- `memory_ports`: sum of all `(ldCount + stCount)` across all memory
  and extmemory instances.
- `temporal_slots`: maximum `num_instruction` across all temporal PEs
  (per-PE instruction capacity). 0 if no temporal PEs exist.

### Constraints

- `module_ref` must reference an existing `fabric.module`.
- `frequency` must be positive.
- Core names must be unique within a `system.design`.

## Operation: `system.link`

A `system.link` represents a point-to-point connection segment in the NoC
between two endpoints (cores or routers).

### Syntax

```mlir
system.link @name {
  src = @source_endpoint,
  dst = @dest_endpoint,
  bandwidth = 4 : i64,
  buffer_depth = 8 : i64,
  latency = 1 : i64,
  type = !dataflow.tagged<!dataflow.bits<32>, i4>
}
```

### Attributes

- `src` (symbol reference): source endpoint (a `system.core` or
  `system.router`).
- `dst` (symbol reference): destination endpoint (a `system.core` or
  `system.router`).
- `bandwidth` (i64): sustained token delivery rate in tokens per cycle.
  This is the primary bandwidth metric used by the analytical model.
  Absolute throughput in bits/second can be derived as
  `bandwidth * token_bitwidth * system_frequency`, where
  `system_frequency` is `min(src.frequency, dst.frequency)` for the
  two endpoints of this link. For links involving routers (which have
  no frequency attribute), the router's effective frequency is inherited
  from the nearest connected core along the path.
- `buffer_depth` (i64): FIFO buffer depth in tokens at the destination end.
- `latency` (i64): minimum traversal latency in cycles (pipeline delay).
- `type`: the payload type carried by this link. Must be
  `!dataflow.bits<N>`, `!dataflow.tagged<!dataflow.bits<N>, iK>`, `memref<...>`,
  or `none`.

### Directionality

Links are unidirectional. A bidirectional connection requires two links.

### Constraints

- `src` and `dst` must reference existing cores or routers within the same
  `system.design`.
- `src` and `dst` must differ (no self-loops).
- `bandwidth` must be >= 1.
- `buffer_depth` must be >= 0 (0 means combinational pass-through).
- `latency` must be >= 0.
- `type` must be a valid link payload type: `!dataflow.bits<N>`,
  `!dataflow.tagged<!dataflow.bits<N>, iK>`, `memref<...>`, or `none`.
- Both endpoints must be type-compatible (same data width on connected
  ports).

## Operation: `system.router`

A `system.router` is an intermediate NoC node that forwards data between
links without performing computation. Routers enable multi-hop routing
between cores that are not directly connected.

### Syntax

```mlir
system.router @name {
  bandwidth = 8 : i64,
  latency = 1 : i64
}
```

### Attributes

- `bandwidth` (i64): maximum aggregate forwarding rate through this router
  in tokens per cycle (shared across all flows traversing it).
- `latency` (i64): per-hop forwarding latency in cycles.

### Semantics

A router is a crossbar-like element: any input link can forward to any
output link, subject to bandwidth constraints. Routers do not buffer data
beyond pipeline latency; buffering is on the links (`buffer_depth`).

Multiple data flows sharing a router degrade per-flow bandwidth
proportionally (fair sharing model in the analytical cost model).

### Constraints

- `bandwidth` must be >= 1.
- `latency` must be >= 0.
- A router must be connected to at least two links (otherwise it is
  unnecessary).
- Router names must be unique within a `system.design`.

## Operation: `system.cache`

A `system.cache` represents an analytical cache node in the memory hierarchy.
It models the performance effect of caching (reduced effective memory latency
for repeated accesses) without defining hardware cache semantics (no fill,
evict, coherence, or replacement policy).

### Syntax

```mlir
system.cache @name {
  capacity = 65536 : i64,
  line_size = 64 : i64,
  associativity = 4 : i64,
  position = @router_or_core,
  access_latency = 2 : i64
}
```

### Attributes

- `capacity` (i64): total cache capacity in bytes.
- `line_size` (i64, optional): cache line size in bytes. Used for reuse
  distance calculations. Default: 64.
- `associativity` (i64, optional): set associativity. Used for conflict
  miss estimation. Default: fully associative (capacity / line_size).
- `position` (symbol reference): the NoC node (router or core) where this
  cache is physically located. Determines NoC distance for spatial
  interaction calculations.
- `access_latency` (i64): cache access latency in cycles (hit case).

### Semantics

A cache node does not intercept or modify data flows in the system dialect.
It exists solely as metadata consumed by the analytical cost model. The cost
model uses cache attributes plus kernel memory access patterns to estimate
hit rates and effective memory latencies.

See [spec-hetero-cache.md](./spec-hetero-cache.md) for the hit rate
estimation model and spatial interaction rules.

### Constraints

- `capacity` must be >= 1.
- `line_size` must be a power of 2 and >= 1.
- `position` must reference an existing core or router.
- `access_latency` must be >= 0.

## Operation: `system.extmem`

A `system.extmem` represents an external memory interface (e.g., a DRAM
controller). It is an analytical abstraction attached to a NoC position
via its `position` attribute, not a vertex in the NoC topology graph.
Memory traffic that misses all caches is routed to the NoC node
specified by `position`. The cost model uses the `extmem` attributes to
compute miss penalties and memory bandwidth limits.

### Syntax

```mlir
system.extmem @name {
  position = @router_or_core,
  latency = 100 : i64,
  bandwidth = 8 : i64
}
```

### Attributes

- `position` (symbol reference): the NoC node (router or core) where
  the external memory interface is located. Determines NoC distance for
  miss-penalty calculations.
- `latency` (i64): external memory access latency in cycles (from
  request arrival at the memory interface to data availability).
- `bandwidth` (i64): sustained bandwidth in tokens per cycle (memory
  controller throughput).

### Semantics

A `system.extmem` node serves as the sink for memory traffic that
is not served by any `system.cache`. The cache effect model uses the NoC
distance from a kernel's assigned core to the nearest `system.extmem` as
the miss path length. When multiple `system.extmem` nodes exist, each
kernel uses the nearest one (by `noc_distance`); ties are broken by
higher bandwidth. See
[spec-hetero-cache.md](./spec-hetero-cache.md).

A `system.design` without any `system.extmem` is structurally valid
(passes IR verification), but the cost model will emit a
`WARN_NO_EXTMEM` diagnostic and skip all memory latency calculations.
In practice, any design with memory-accessing kernels should include at
least one `system.extmem`.

### Constraints

- `position` must reference an existing core or router.
- `latency` must be >= 1.
- `bandwidth` must be >= 1.
- Names must be unique within a `system.design`.

## Type System Integration

The system dialect uses types from two sources:

**Dataflow dialect types** (see [spec-fabric.md](./spec-fabric.md) for
fabric-level usage):
- `!dataflow.bits<N>` for untagged data payloads
- `!dataflow.tagged<!dataflow.bits<N>, iK>` for tagged streaming data
- `none` for control-only tokens

**Standard MLIR types**:
- `memref<...>` for MMIO memory-mapped interfaces

No new types are introduced by the system dialect. These types are used
as link payload types and kernel port types.

## Relationship to Fabric Dialect

The system dialect and fabric dialect have a strict containment relationship:

```
system.design
  system.core @c0 { module_ref = @my_cgra }
                                    |
                                    v
                          fabric.module @my_cgra { ... }
```

Rules:
- `system.core` references `fabric.module` by symbol.
- The system dialect never contains fabric operations directly.
- Fabric operations never contain system operations.
- A `fabric.module` may be referenced by multiple `system.core` instances
  (homogeneous cores share the same hardware template).
- Modifying a `fabric.module` does not require changes to the system
  dialect (separation of concerns).

## Serialization

System dialect operations serialize to MLIR textual format following
standard MLIR conventions. A system design file uses the `.system.mlir`
extension by convention.

A complete system description requires both the system design file and the
referenced fabric module files to be available.

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-kernel.md](./spec-hetero-kernel.md)
- [spec-hetero-noc.md](./spec-hetero-noc.md)
- [spec-hetero-cache.md](./spec-hetero-cache.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-dataflow.md](./spec-dataflow.md)

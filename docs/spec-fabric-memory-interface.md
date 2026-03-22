# LOOM Fabric Memory Interface Specification

## Overview

LOOM memory-facing hardware uses explicit external-memory resources together with
switch routing, optional tag transforms, and a backend-independent memory
backing abstraction.

Placement rules:

- `fabric.memory` and `fabric.extmemory` definitions may appear directly in the
  top-level module or in `fabric.module`
- inline `fabric.memory` and `fabric.extmemory` instantiations may appear
  directly only in `fabric.module`
- `fabric.instance` targeting one `fabric.memory` or `fabric.extmemory`
  definition may appear directly only in `fabric.module`

## Memory-Interface Placement Rule

LOOM does not require a one-to-one relationship between software memrefs and
hardware memory interfaces.

Instead:

- each software `handshake.extmemory` or `handshake.memory` is placed onto a
  compatible hardware `fabric.extmemory` or `fabric.memory`
- one hardware memory interface may host multiple software memory regions up to
  its `numRegion` capacity
- region selection is part of mapping, not a fixed syntactic binding

This means the mapper may choose either:

- separate hardware memory interfaces for separate software regions
- or a shared hardware memory interface with multiple regions

## Single-Port vs Multi-Port Memory

When load and store counts are both one, a memory may be used without tagged
memory routing.

When `ldCount > 1` or `stCount > 1`, LOOM uses tagged memory routing so multiple
logical streams can share one memory-facing endpoint.

## Hardware Interface Families

LOOM hardware memory interfaces are organized by signal family, not by software
operand order.

`fabric.memory` uses these physical families:

- inputs: `load_addr`, `store_addr`, `store_data`
- outputs: `load_data`, `load_done`, `store_done`

Family omission is structural:

- if `ldCount = 0`, there is no `load_addr`, `load_data`, or `load_done`
- if `stCount = 0`, there is no `store_addr`, `store_data`, or `store_done`

When `ldCount > 1` or `stCount > 1`, the family still appears once
physically, but its payload becomes tagged.

For a tagged hardware memory-family port:

- tagged versus non-tagged is a hardware parameter
- `tagWidth` is a hardware parameter
- the concrete tag carried by one software stream is a runtime value

For `fabric.memory` and `fabric.extmemory`, the tagged family width must be
large enough to encode the maximum logical lane count:

- `tagWidth >= log2Ceil(max(ldCount, stCount))`

The mapper may assign or transform runtime tag values, but it does not infer
the hardware tag width.

`fabric.memory` may additionally expose a memref-style externally visible view
when `is_private` is not true. That memref represents a slave-style,
memory-mapped access path into the scratchpad. In the visualization model,
that public memref is the opposite-facing counterpart of `fabric.extmemory`:

- `fabric.extmemory` shows one memref input on the ingress side
- non-private `fabric.memory` shows one memref output on the egress side

`fabric.extmemory` uses the same request and response family order, but always
consumes one incoming module memref operand first. That memref represents the
master-style backing memory interface that the accelerator actively accesses.

## Software Memory-Op Order

Software memory ops use CIRCT Handshake ordering, which is intentionally
different from hardware family order.

`handshake.memory` uses:

- operands: all stores first as `(stdata1, staddr1, stdata2, staddr2, ...)`,
  then all loads as `(ldaddr1, ldaddr2, ...)`
- results: all load data as `(lddata1, lddata2, ...)`, then all completion
  tokens ordered like the request operands:
  `(stnone1, stnone2, ..., ldnone1, ldnone2, ...)`

`handshake.extmemory` uses the same load and store ordering, with one leading
memref operand naming the backing memory object.

The mapper must therefore bridge between:

- software request order: store-first, then load addresses
- hardware family order: load address first, then store address/data
- software result order: load data, then store done, then load done
- hardware family order: load data, then load done, then store done

## Tagged Multi-Port Memory Mechanism

For multi-port memory access:

1. each load or store stream is tagged with its logical port index
2. tagged streams are merged through routing fabric into the memory endpoint
3. the memory demultiplexes by tag internally
4. return data or completion tokens preserve tag identity on the way back

This mechanism does not require one fixed micro-topology.

The essential requirement is:

- software streams that share one hardware tagged memory-family port must carry
  distinct runtime tag values along the shared portion of the path

Those runtime tag values may be introduced or transformed in multiple ways:

- by `fabric.add_tag`
- by `fabric.map_tag`
- by a hierarchy of tagged routing stages

They may be merged:

- through one centralized tagged switch
- or through multiple staged tagged switches

They may be stripped on egress by `fabric.del_tag`, but stripping is only
required when the destination side expects a non-tagged value.

In other words, LOOM does not require a canonical chain such as:

- `add_tag -> one switch -> memory`
- or `memory -> one switch -> del_tag`

Bridge extraction for mapper and visualization purposes may therefore stop at
the nearest tagged route-stage port that bounds the shared memory path, even
when no explicit `fabric.add_tag` or `fabric.del_tag` exists at the compute
container side. This is valid when the compute-facing side already remains
tagged.

For conflict checking and route validation, LOOM must compare software streams
on the full bridge-expanded shared path, not only on the truncated
place-and-route boundary path stored in mapper state. Shared tagged resources
may appear entirely inside the recovered bridge suffix or prefix.

What matters is the semantic contract:

- a shared tagged family port must see distinct runtime tag values for the
  logically different software streams that share it
- any `fabric.map_tag` on that path changes the runtime tag value seen by later
  tag-dependent routing or execution stages
- operations other than `fabric.add_tag`, `fabric.map_tag`, and
  `fabric.del_tag` do not change tagged shape; they only transport it

## Relationship to Switch Semantics

Tagged memory traffic still obeys the switch rules of the enclosing routing
resources. The only difference is that payload identity includes a tag field.

LOOM may carry tagged memory traffic through:

- `fabric.temporal_sw`, when route choice is tag-dependent
- `fabric.spatial_sw`, when route choice is tag-agnostic and the tag is only
  part of the payload

In current LOOM memory-bridge design intent:

- ingress request mixing may use any tagged, tag-compatible routing path
- tagged `fabric.spatial_sw` is legal for tag-agnostic merging
- tagged `fabric.temporal_sw` is legal for tag-dependent merging or splitting
- ingress may use one stage or multiple hierarchical stages of tagged routing
- if an ingress path already carries a tag before it reaches the memory-family
  port, that path-derived tag remains authoritative
- the software lane id is only a fallback runtime-tag source when the routed
  path has not attached any tag yet
- egress may apply `fabric.map_tag` before the tag-dependent split if the
  response tag namespace must be rewritten
- egress separation may keep the tag when the destination remains tagged
- egress separation uses `fabric.del_tag` only when the destination expects a
  non-tagged value

The memory tagging scheme does not authorize illegal structural merging at one
spatial-switch output, whether the spatial switch payload is tagged or
non-tagged.

This remains true after tag-width adaptation. Two source-side tag values that
start out different may still be illegal if their routed path would require one
of them to become unrepresentable on a tagged hardware port, or if an explicit
later tag transform makes them equal on a shared tagged resource.

LOOM must not treat implicit width adaptation itself as a legal tag-rewrite
mechanism. Only explicit `fabric.add_tag`, `fabric.map_tag`, or
`fabric.del_tag` boundaries may change the runtime tag meaning seen by later
shared tagged resources.

## Memory Region Model

LOOM treats external memory as a set of numbered regions. Each region has:

- a region id
- a tag range
- a base address
- an element-size code
- a backing implementation

The accelerator runtime binds these regions before launch.

`addr_offset_table` entries use the LOOM tuple:

- `valid`
- `start_tag`
- `end_tag`
- `addr_offset`
- `elem_size_log2`

`elem_size_log2` follows AXI `AxSIZE` style encoding:

- `0` = 1 byte
- `1` = 2 bytes
- `2` = 4 bytes
- `3` = 8 bytes

The element-size code is per region because different software memory regions
may share one wider hardware memory interface.

This region model works together with tagged family ports:

- `addr_offset_table` identifies which tag range belongs to which software
  region, where that region starts, and what element size it uses
- tagged request and response ports carry the stream identity for that region
  or logical access lane
- different tags may progress independently
- requests with the same tag must preserve order

In LOOM syntax:

- hardware-structure fields such as `ldCount`, `stCount`, `lsqDepth`,
  `memrefType`, and `numRegion` belong to `[]`
- runtime region programming such as `addr_offset_table` belongs to
  `attributes {}`

One important LOOM distinction is ownership of the region base address:

- for `fabric.memory`, the mapper may compute on-chip base offsets directly
- for `fabric.extmemory`, the mapper does not know the host virtual or physical
  base address of the backing memory object

Therefore:

- `fabric.extmemory` configuration emitted by the mapper always serializes
  `addr_offset` as `0`
- the host runtime is responsible for patching or programming the actual
  base address before launch

## Memref Type Compatibility

For software-to-hardware memory mapping, LOOM matches memrefs by element width,
not by the original scalar kind.

Examples:

- `handshake.extmemory(memref<?xi32>)` may map to
  `fabric.extmemory(memref<?xi32>)`
- `handshake.extmemory(memref<?xf32>)` may also map to
  `fabric.extmemory(memref<?xi32>)`
- `handshake.extmemory(memref<?xi16>)` may map to
  `fabric.extmemory(memref<?xi64>)`

The rule is:

- software memref element width must be less than or equal to hardware memory
  interface element width
- exact integer-vs-float element kind is not relevant at mapping time

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

# FCC Fabric Memory Interface Specification

## Overview

FCC memory-facing hardware uses explicit external-memory resources together with
switch routing, optional tag transforms, and a backend-independent memory
backing abstraction.

## Memory-Interface Placement Rule

FCC does not require a one-to-one relationship between software memrefs and
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

When `ldCount > 1` or `stCount > 1`, FCC uses tagged memory routing so multiple
logical streams can share one memory-facing endpoint.

## Hardware Interface Families

FCC hardware memory interfaces are organized by signal family, not by software
operand order.

`fabric.memory` uses these physical families:

- inputs: `load_addr`, `store_addr`, `store_data`
- outputs: `load_data`, `load_done`, `store_done`

Family omission is structural:

- if `ldCount = 0`, there is no `load_addr`, `load_data`, or `load_done`
- if `stCount = 0`, there is no `store_addr`, `store_data`, or `store_done`

When `ldCount > 1` or `stCount > 1`, the family still appears once
physically, but its payload becomes tagged.

`fabric.memory` may additionally expose a memref-style externally visible view
when `is_private` is not true. That memref represents a slave-style,
memory-mapped access path into the scratchpad.

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
- a tag range
- a base address
- an element-size code
- a backing implementation

The accelerator runtime binds these regions before launch.

`addr_offset_table` entries use the FCC tuple:

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

In FCC syntax:

- hardware-structure fields such as `ldCount`, `stCount`, `lsqDepth`,
  `memrefType`, and `numRegion` belong to `[]`
- runtime region programming such as `addr_offset_table` belongs to
  `attributes {}`

## Memref Type Compatibility

For software-to-hardware memory mapping, FCC matches memrefs by element width,
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

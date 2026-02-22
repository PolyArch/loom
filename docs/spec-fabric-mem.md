# Fabric Memory Specification

## Overview

This document specifies the fabric memory operations:

- `fabric.memory`: an on-chip scratchpad memory block.
- `fabric.extmemory`: an external memory interface.

Both operations are hardware-facing and are intended to map directly to RTL
modules. They are the fabric-level counterparts of `handshake.memory` and
`handshake.extmemory`.

Compile-time, configuration-time, and runtime error codes are defined in
[spec-fabric-error.md](./spec-fabric-error.md).

## Read-Only Memory Mapping (ROM)

Loom does not define a separate `fabric.rom` operation. Read-only memory is
represented using existing memory operations with no store ports:

- On-chip read-only memory: `fabric.memory` with `stCount = 0`
- External read-only memory: `fabric.extmemory` with `stCount = 0`

The source object's constant initializer is used to initialize memory contents.
This representation is the lowering target for `LOOM_TARGET("rom")` hints.
See [spec-pragma.md](./spec-pragma.md).

## Operation: `fabric.memory` and `fabric.extmemory`

### Forms

`fabric.memory` and `fabric.extmemory` support two forms:

- **Named form**: defines a reusable memory module with a symbol name.
- **Inline form**: defines a local memory used directly in the surrounding
  region.

Both forms share the same semantics and constraints. Named memory modules can
be instantiated via `fabric.instance`.

### Named Form Syntax

```mlir
fabric.memory @scratchpad
    [ldCount = 2, stCount = 2, lsqDepth = 4, is_private = true]
    : memref<1024xi32>,
      (!dataflow.tagged<index, i2>, !dataflow.tagged<index, i2>,
       !dataflow.tagged<i32, i2>)
      -> (!dataflow.tagged<i32, i2>,
          !dataflow.tagged<none, i2>,
          !dataflow.tagged<none, i2>)

fabric.extmemory @dram_if
    [ldCount = 1, stCount = 1, lsqDepth = 4]
    : memref<?xf32>, (index, index, f32) -> (f32, none, none)
```

Named memory modules can be instantiated with `fabric.instance`:

```mlir
%lddata, %lddone, %stdone = fabric.instance @scratchpad(%ldaddr, %staddr, %stdata)
    : (!dataflow.tagged<index, i2>, !dataflow.tagged<index, i2>,
       !dataflow.tagged<i32, i2>)
      -> (!dataflow.tagged<i32, i2>,
          !dataflow.tagged<none, i2>,
          !dataflow.tagged<none, i2>)
```

### Inline Form Syntax

The following syntax sketches the required structure. The operation may omit
the load or store group when `ldCount == 0` or `stCount == 0`.

Single-port example:

```mlir
%lddata, %lddone, %stdone = fabric.memory
    [ldCount = 1, stCount = 1, lsqDepth = 4, is_private = true]
    (%ldaddr, %staddr, %stdata)
    : memref<256xi32>, (index, index, i32) -> (i32, none, none)
```

Multi-port example with tags:

```mlir
%lddata, %lddone, %stdone = fabric.extmemory
    [ldCount = 4, stCount = 2, lsqDepth = 8]
    (%memref, %ldaddr, %staddr, %stdata)
    : memref<?xf32>,
      (!dataflow.tagged<index, i3>,
       !dataflow.tagged<index, i3>,
       !dataflow.tagged<f32, i3>)
      -> (!dataflow.tagged<f32, i3>,
          !dataflow.tagged<none, i3>,
          !dataflow.tagged<none, i3>)
```

Memory export example:

```mlir
%mem, %lddata, %lddone = fabric.memory
    [ldCount = 1, stCount = 0, is_private = false]
    (%ldaddr)
    : memref<1024xi16>, (index) -> (memref<1024xi16>, i16, none)

fabric.yield %mem : memref<1024xi16>
```

### Attributes

All attributes in this section are hardware parameters.

- `ldCount`: number of logical load ports.
- `stCount`: number of logical store ports.
- `lsqDepth`: load-store queue (LSQ) depth (only meaningful when `stCount > 0`).
- `is_private` (only for `fabric.memory`): if `true`, the memory is private to
  the module. If `false`, the memory exposes an output memref that must be
  yielded by `fabric.module`. The default is `true`.
- `numRegion`: number of entries in the addr_offset_table. Must be >= 1.
  The default is `1`. Each region entry contains a valid bit, start_tag,
  end_tag, and addr_offset. At runtime, a load/store tag is matched against
  the [start_tag, end_tag) half-open range of each valid region, and the corresponding
  addr_offset is added to the memory address. CONFIG_WIDTH for a memory is
  `numRegion * (1 + 2 * TAG_WIDTH + ADDR_WIDTH)`.
- `fabric.extmemory` does not support `is_private`. Supplying `is_private` on
  `fabric.extmemory` is invalid (`CPL_MEMORY_EXTMEM_PRIVATE`).

### Port Groups

Load group:

- Input: `ldaddr`
- Outputs: `lddata`, `lddone`

Store group:

- Inputs: `staddr`, `stdata`
- Output: `stdone`

The presence of a group depends on `ldCount` and `stCount`. If a group is
absent, its ports do not exist.

For `fabric.extmemory`, the first operand is always the memref input that binds
to a `fabric.module` memref input port. For `fabric.memory`, there is no memref
operand.

### Type Rules

Address and data types:

- `ldaddr` and `staddr` must be `index`, or `!dataflow.tagged<index, iK>`.
- `lddata` and `stdata` value types must match the memref element type.
- When tagged, `lddata`, `stdata`, and the corresponding addr must share the
  same tag width.

Done tokens:

- If `ldCount == 1`, `lddone` is `none`.
- If `ldCount > 1`, `lddone` is `!dataflow.tagged<none, iK>` and carries the
  same tag as the corresponding `ldaddr`.
- If `stCount == 1`, `stdone` is `none`.
- If `stCount > 1`, `stdone` is `!dataflow.tagged<none, iK>` and carries the
  same tag as the corresponding store request.

**IMPORTANT: Breaking change when modifying port counts.** Changing `ldCount`
from 1 to 2 (or vice versa) changes the `lddone` type from `none` to
`!dataflow.tagged<none, iK>`. Similarly, changing `stCount` from 1 to 2 changes
`stdone` type. This is a breaking interface change that requires updating all
consumers of the done token.

### Tagging Rules

- If either `ldCount > 1` or `stCount > 1`, ALL ports of the memory instance
  must be tagged with `TAG_WIDTH = clog2(max(ldCount, stCount))`.
- If both `ldCount <= 1` and `stCount <= 1`, no ports may be tagged
  (`TAG_WIDTH = 0`).

Tags identify the logical port. The valid tag range is `[0, count - 1]`. A tag
value outside this range is a runtime error (`RT_MEMORY_TAG_OOB`).

### Ordering Rules

For both load and store:

- Requests with the same tag are in-order.
- Requests with different tags may complete out of order.
- `lddone` aligns with `lddata`.
- `stdone` aligns with the completion of the corresponding store request.

**Same-cycle load/store collision.** When a load and a store target the same
element address in the same cycle, the load returns the value that existed at
the beginning of the cycle (read-before-write semantics). This applies to both
`fabric.memory` and `fabric.extmemory`. The dataflow compiler is responsible
for inserting proper control tokens to enforce any required read-after-write
ordering across cycles.

### Store Queue Semantics

`lsqDepth` defines the capacity of the internal store queue used to pair
`staddr` and `stdata`.

- If `stCount == 0`, `lsqDepth` must be `0`.
- If `stCount > 0`, `lsqDepth` must be >= 1.

**Multi-port (tagged, `stCount > 1`):**

Stores are paired per tag in FIFO order. For each tag, the queue matches
`staddr` and `stdata` arrivals in the order they are received.

**Single-port (native, `stCount == 1`):**

Since there is only one logical store port (native, no tags), the queue pairs
`staddr` and `stdata` in strict FIFO order. The first `staddr` matches with
the first `stdata`, the second with the second, and so on.

**Deadlock detection:**

For each store tag (or the single global store queue in native mode), the
hardware maintains a deadlock counter. The counter behavior is:

- **Imbalance condition**: exactly one of `staddr` or `stdata` has a queued
  entry while the other side is empty for that tag.
- **Counting**: the counter increments by 1 each cycle while the imbalance
  condition holds continuously. If the imbalance condition ceases (both sides
  have data, or both sides are empty), the counter resets to 0 immediately.
- **Trigger**: `RT_MEMORY_STORE_DEADLOCK` is raised after the counter has
  incremented for exactly `DEADLOCK_TIMEOUT` consecutive cycles under the
  imbalance condition.
- **Default timeout**: 65535 cycles (see
  [spec-fabric-error.md](./spec-fabric-error.md)).

This rule applies identically to `fabric.memory` and `fabric.extmemory`.
When multiple error conditions are true in the same cycle, the error with the
numerically smallest code is captured. See
[spec-fabric-error.md](./spec-fabric-error.md) for the cross-module
precedence rule.

### Memory Export (`fabric.memory` only)

`fabric.memory` may expose an output memref when `is_private = false`. The output
memref is the first result of the operation, must be yielded by
`fabric.module`, and appears in the module result list.

If a `fabric.module` yields a memref that is not produced by a non-private
`fabric.memory`, the compiler must raise `CPL_MEMORY_PRIVATE_OUTPUT`.

`fabric.extmemory` never produces a memref output.

### Address Semantics and Data Width

The address ports use element indices, not byte addresses. The hardware
converts element indices into byte addresses using the memref element size.

The memref element type defines the fixed access width of the memory interface.
For example, `memref<i64>` implies an 8-byte access width for each load or
store. A memory interface cannot be used to access a different element width.

### Static vs Dynamic Shapes

- `fabric.extmemory` may use dynamic memref shapes.
- `fabric.memory` must use a static memref shape. If the source IR does not
  provide a fixed size, the compiler must select a fixed capacity. The default
  policy is 4KB, and the toolchain may override this default.

### Constraints

Violations of the following constraints are compile-time errors:

- `ldCount == 0` and `stCount == 0` (`CPL_MEMORY_PORTS_EMPTY`).
- `lsqDepth != 0` when `stCount == 0` (`CPL_MEMORY_LSQ_WITHOUT_STORE`).
- `lsqDepth < 1` when `stCount > 0` (`CPL_MEMORY_LSQ_MIN`).
- Address ports are not `index` or tagged `index` (`CPL_MEMORY_ADDR_TYPE`).
- Data value type does not match memref element type (`CPL_MEMORY_DATA_TYPE`).
- Tagging requirements are not met (`CPL_MEMORY_TAG_REQUIRED`,
  `CPL_MEMORY_TAG_FOR_SINGLE`, `CPL_MEMORY_TAG_WIDTH`).
- Dynamic memref shape on `fabric.memory` (`CPL_MEMORY_STATIC_REQUIRED`).
- `fabric.extmemory` memref operand is not a module memref input
  (`CPL_MEMORY_EXTMEM_BINDING`).
- `is_private` is supplied on `fabric.extmemory`
  (`CPL_MEMORY_EXTMEM_PRIVATE`).
- `numRegion < 1` (`CPL_MEMORY_INVALID_REGION`).

Runtime configuration errors (detected in hardware):

- Overlapping tag ranges between valid regions in addr_offset_table
  (`CFG_MEMORY_OVERLAP_TAG_REGION`, `CFG_EXTMEMORY_OVERLAP_TAG_REGION`).
- A region has `end_tag <= start_tag` (empty half-open range)
  (`CFG_MEMORY_EMPTY_TAG_RANGE`, `CFG_EXTMEMORY_EMPTY_TAG_RANGE`).

Runtime execution errors (detected in hardware):

- A load/store tag matches no valid region in the addr_offset_table
  (`RT_MEMORY_NO_MATCH`, `RT_EXTMEMORY_NO_MATCH`).

## Interaction with Load/Store PEs

Load/store PEs are adapters between compute and `fabric.memory` /
`fabric.extmemory`. They perform synchronization and tag handling, while the
memory operation performs the actual memory access. See
[spec-fabric-pe.md](./spec-fabric-pe.md) for load/store PE semantics.

See [spec-fabric.md](./spec-fabric.md) Operation Syntax Conventions for the
canonical `[hw_params] {runtime_config}` bracket convention.

## Port Width by Category

Memory module ports have category-specific widths:

| Port Category | Data Width | Notes |
|---------------|-----------|-------|
| `ld_addr[i]` | 64 bits (index type) | Fixed address width |
| `st_addr[i]` | 64 bits (index type) | Fixed address width |
| `st_data[i]` | `elemType` width | Element data width |
| `ld_data[i]` | `elemType` width | Element data width |
| `ld_done[i]` | 0 bits (none type) | Completion signal only |
| `st_done[i]` | 0 bits (none type) | Completion signal only |

`TAG_WIDTH` is uniform across all ports of a memory instance:
`TAG_WIDTH = clog2(max(LD_COUNT, ST_COUNT))` when either count > 1, else 0.
(`clog2` = ceiling of log base 2.)

When untagged (`TAG_WIDTH = 0`), done ports use a 1-bit minimum for hardware
positive width. When tagged, done port width = `TAG_WIDTH` bits (none carries
0 data bits).

**Example**: Memory with `elemType=i32`, `LD_COUNT=2`, `ST_COUNT=1`:

```
TAG_WIDTH = clog2(max(2,1)) = 1
ld_addr0: 64 + 1 = 65 bits payload
st_addr0: 64 + 1 = 65 bits payload
st_data0: 32 + 1 = 33 bits payload
ld_data0: 32 + 1 = 33 bits payload
ld_done0: 0 + 1  = 1  bit payload (none=0 data bits + TAG_WIDTH)
st_done0: 0 + 1  = 1  bit payload (none=0 data bits + TAG_WIDTH)
```

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-error.md](./spec-fabric-error.md)

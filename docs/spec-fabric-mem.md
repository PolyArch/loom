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
    [ldCount = 2, stCount = 2, lsqDepth = 4, private = true]
    : memref<1024xi32>,
      (!dataflow.tagged<index, i2>, !dataflow.tagged<index, i2>,
       !dataflow.tagged<i32, i2>)
      -> (!dataflow.tagged<i32, i2>,
          !dataflow.tagged<i1, i2>,
          !dataflow.tagged<i1, i2>)

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
          !dataflow.tagged<i1, i2>,
          !dataflow.tagged<i1, i2>)
```

### Inline Form Syntax

The following syntax sketches the required structure. The operation may omit
the load or store group when `ldCount == 0` or `stCount == 0`.

Single-port example:

```mlir
%lddata, %lddone, %stdone = fabric.memory
    [ldCount = 1, stCount = 1, lsqDepth = 4, private = true]
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
          !dataflow.tagged<i1, i3>,
          !dataflow.tagged<i1, i3>)
```

Memory export example:

```mlir
%mem, %lddata, %lddone = fabric.memory
    [ldCount = 1, stCount = 0, private = false]
    (%ldaddr)
    : memref<1024xi16>, (index) -> (memref<1024xi16>, i16, none)

fabric.yield %mem : memref<1024xi16>
```

### Attributes

All attributes in this section are hardware parameters.

- `ldCount`: number of logical load ports.
- `stCount`: number of logical store ports.
- `lsqDepth`: store queue depth (only meaningful when `stCount > 0`).
- `private` (only for `fabric.memory`): if `true`, the memory is private to the
  module. If `false`, the memory exposes an output memref that must be yielded
  by `fabric.module`. The default is `true`.

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
- If `ldCount > 1`, `lddone` is `!dataflow.tagged<i1, iK>` and carries the same
  tag as the corresponding `ldaddr`. The `i1` payload is a dummy constant `0`.
  See [spec-dataflow.md](./spec-dataflow.md).
- If `stCount == 1`, `stdone` is `none`.
- If `stCount > 1`, `stdone` is `!dataflow.tagged<i1, iK>` and carries the same
  tag as the corresponding store request. The `i1` payload is a dummy constant
  `0`. See [spec-dataflow.md](./spec-dataflow.md).

**IMPORTANT: Breaking change when modifying port counts.** Changing `ldCount`
from 1 to 2 (or vice versa) changes the `lddone` type from `none` to
`!dataflow.tagged<i1, iK>`. Similarly, changing `stCount` from 1 to 2 changes
`stdone` type. This is a breaking interface change that requires updating all
consumers of the done token.

### Tagging Rules

- If `ldCount > 1`, `ldaddr` and `lddata` must be tagged.
- If `stCount > 1`, `staddr` and `stdata` must be tagged.
- If `ldCount == 1`, tagged load ports are not allowed.
- If `stCount == 1`, tagged store ports are not allowed.
- Tag width `K` must satisfy `K >= log2Ceil(count)` for each group.

Tags identify the logical port. The valid tag range is `[0, count - 1]`. A tag
value outside this range is a runtime error (`RT_MEMORY_TAG_OOB`).

### Ordering Rules

For both load and store:

- Requests with the same tag are in-order.
- Requests with different tags may complete out of order.
- `lddone` aligns with `lddata`.
- `stdone` aligns with the completion of the corresponding store request.

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

For both tagged and native modes, if a store request cannot be matched because
one side (address or data) is missing and no progress is possible, hardware
reports `RT_MEMORY_STORE_DEADLOCK` after a configurable timeout. This applies
equally to the native single-port case (e.g., `staddr` waiting for `stdata`
or vice versa). The default timeout is defined in
[spec-fabric-error.md](./spec-fabric-error.md).

### Memory Export (`fabric.memory` only)

`fabric.memory` may expose an output memref when `private = false`. The output
memref is the first result of the operation, must be yielded by
`fabric.module`, and appears in the module result list.

If a `fabric.module` yields a memref that is not produced by a non-private
`fabric.memory`, the compiler must raise `COMP_MEMORY_PRIVATE_OUTPUT`.

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

- `ldCount == 0` and `stCount == 0` (`COMP_MEMORY_PORTS_EMPTY`).
- `lsqDepth != 0` when `stCount == 0` (`COMP_MEMORY_LSQ_WITHOUT_STORE`).
- `lsqDepth < 1` when `stCount > 0` (`COMP_MEMORY_LSQ_MIN`).
- Address ports are not `index` or tagged `index` (`COMP_MEMORY_ADDR_TYPE`).
- Data value type does not match memref element type (`COMP_MEMORY_DATA_TYPE`).
- Tagging requirements are not met (`COMP_MEMORY_TAG_REQUIRED`,
  `COMP_MEMORY_TAG_FOR_SINGLE`, `COMP_MEMORY_TAG_WIDTH`).
- Dynamic memref shape on `fabric.memory` (`COMP_MEMORY_STATIC_REQUIRED`).
- `fabric.extmemory` memref operand is not a module memref input
  (`COMP_MEMORY_EXTMEM_BINDING`).

## Interaction with Load/Store PEs

Load/store PEs are adapters between compute and `fabric.memory` /
`fabric.extmemory`. They perform synchronization and tag handling, while the
memory operation performs the actual memory access. See
[spec-fabric-pe.md](./spec-fabric-pe.md) for load/store PE semantics.

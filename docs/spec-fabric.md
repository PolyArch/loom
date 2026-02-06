# Fabric Dialect Specification

## Overview

The fabric dialect is the hardware IR for Loom. It models a full accelerator as
interconnected hardware modules with explicit streaming and memory interfaces.
The dialect is designed to be a mapping target for software dataflow graphs.
For mapper semantics (place-and-route from Handshake/Dataflow graphs), see
[spec-mapper.md](./spec-mapper.md).

This document defines the fabric dialect at a top level and provides full
specifications for `fabric.module`, `fabric.instance`, and `fabric.yield`. Other
operations are specified in dedicated documents referenced below.

## Operation Summary

| Category | Operations | Details |
|----------|------------|---------|
| Top level | `fabric.module` | This document |
| Instantiation | `fabric.instance` | This document |
| Terminator | `fabric.yield` | This document |
| Processing | `fabric.pe`, `fabric.temporal_pe` | [spec-fabric-pe.md](./spec-fabric-pe.md), [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md), [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) |
| Routing | `fabric.switch`, `fabric.temporal_sw` | [spec-fabric-switch.md](./spec-fabric-switch.md), [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md) |
| Tag boundary | `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag` | [spec-fabric-tag.md](./spec-fabric-tag.md) |
| Memory | `fabric.memory`, `fabric.extmemory` | [spec-fabric-mem.md](./spec-fabric-mem.md) |

## Type Conventions

The fabric dialect uses the dataflow tagged type for temporal routing and
streaming:

- `!dataflow.tagged<value_type, tag_type>`

See [spec-dataflow.md](./spec-dataflow.md) for the tagged type definition.

Within fabric operations, the term **native value type** follows the exact value
type set defined in [spec-dataflow.md](./spec-dataflow.md). Vector, tensor,
memref, complex, and opaque types are not considered native value types.

The **none type** represents a control-only token with valid/ready signals but
no data payload. It is used for synchronization tokens such as `ctrl` ports on
load/store PEs and single-port `lddone`/`stdone` ports on memory operations.
For multi-port tagged memories, done tokens use `!dataflow.tagged<none, iK>`.
See [spec-dataflow.md](./spec-dataflow.md) and
[spec-fabric-mem.md](./spec-fabric-mem.md) for the authoritative rule.

## Hardware Parameters vs Runtime Configuration

Many fabric operations split parameters into two categories:

- **Hardware parameters** describe physical structure and are fixed for a given
  hardware instance.
- **Runtime configuration parameters** are stored in hardware registers or
  tables and may be reprogrammed at runtime.

Each operation document explicitly identifies which attributes are hardware
parameters and which are runtime configuration parameters.

## Operation: `fabric.module`

A `fabric.module` represents an accelerator as a hardware module with explicit
streaming and memory ports.

### Syntax

```
fabric.module @name(
  %mem_in_0: memref<...>, ...,
  %native_in_0: T, ...,
  %tagged_in_0: !dataflow.tagged<...>, ...
) -> (
  memref<...>, ...,
  T, ...,
  !dataflow.tagged<...>, ...
) {
  // body
  fabric.yield %mem_out_0, ..., %native_out_0, ..., %tagged_out_0, ...
}
```

### Port Ordering (Required)

The argument and result ordering is fixed and must be preserved:

- Inputs: `memref*`, `native*`, `tagged*`
- Outputs: `memref*`, `native*`, `tagged*`

### Port Categories and Semantics

- **Memref inputs (M ports)**
  - Represent active master interfaces that access external memory.
  - Hardware semantics are similar to an AXI Master interface.

- **Memref outputs (I ports)**
  - Represent passive slave interfaces that expose internal memory to the
    outside.
  - Hardware semantics are similar to an AXI Slave interface.

- **Native value inputs (N ports)**
  - Passive streaming inputs, similar to an AXI-Stream slave interface.

- **Tagged inputs (O ports)**
  - Passive streaming inputs carrying `!dataflow.tagged` values.

- **Native value outputs (J ports)**
  - Active streaming outputs, similar to an AXI-Stream master interface.

- **Tagged outputs (K ports)**
  - Active streaming outputs carrying `!dataflow.tagged` values.

### Constraints

- `M + N + O + I + J + K` must be greater than 0.
  - An accelerator cannot be a completely empty shell.
- All `tagged` ports must use valid `!dataflow.tagged` types.
- Port ordering must follow: memref*, native*, tagged* for both inputs and
  outputs. Violations raise `COMP_MODULE_PORT_ORDER`.
- The body must contain at least one non-terminator operation. Violations
  raise `COMP_MODULE_EMPTY_BODY`.
- The body must end with `fabric.yield`. Violations raise
  `COMP_MODULE_MISSING_YIELD`.

See [spec-fabric-error.md](./spec-fabric-error.md) for error code definitions.

### Body and Terminator

The body of `fabric.module` contains a hardware graph built from fabric
operations. Only fabric operations are allowed at the module level.

Allowed operations at module level:

- `fabric.pe`
- `fabric.temporal_pe`
- `fabric.switch`
- `fabric.temporal_sw`
- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`
- `fabric.memory`
- `fabric.extmemory`
- `fabric.instance`
- `fabric.yield`

Non-fabric operations are not allowed directly at module level. Operations
inside `fabric.pe` must follow the authoritative allowlist in
[spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md).

The body must end with `fabric.yield` and the yielded values must match the
module result types and ordering.
Any memref result must originate from a `fabric.memory` with
`is_private = false`. See [spec-fabric-mem.md](./spec-fabric-mem.md).

### Explicit Type Matching

All connections inside a `fabric.module` must match exactly in type and bit
width. There are no implicit width extensions, truncations, or tag
conversions. Any type mismatch on a connection is a compile-time error.
(`COMP_FABRIC_TYPE_MISMATCH`; see [spec-fabric-error.md](./spec-fabric-error.md).)
Explicit conversions must be represented as operations, such as:

- `fabric.add_tag`, `fabric.del_tag`, and `fabric.map_tag` for tag boundaries or
  tag transforms.
- `fabric.pe` containing explicit casts (e.g., `arith.index_cast`).

## Operation: `fabric.instance`

Instantiates a named fabric module or hardware component.

### Syntax

```
%res0, %res1 = fabric.instance @target(%arg0, %arg1)
  {sym_name = "u0"} : (T0, T1) -> (R0, R1)
```

### Attributes

- `module` (implicit): symbol reference to the target operation.
- `sym_name`: string name for the instance.

### Constraints

- The referenced symbol must be one of:
  - `fabric.module`
  - `fabric.pe`
  - `fabric.temporal_pe`
  - `fabric.switch`
  - `fabric.temporal_sw`
  - `fabric.memory`
  - `fabric.extmemory`
- `fabric.add_tag`, `fabric.map_tag`, and `fabric.del_tag` cannot be
  instantiated. They must be used inline. Violations raise
  `COMP_INSTANCE_ILLEGAL_TARGET`.
- The referenced symbol must exist. Violations raise
  `COMP_INSTANCE_UNRESOLVED`.
- Operand count and types must match the referenced module signature.
  Violations raise `COMP_INSTANCE_OPERAND_MISMATCH`.
- Result count and types must match the referenced module signature.
  Violations raise `COMP_INSTANCE_RESULT_MISMATCH`.
- Scope-specific restrictions still apply:
  - Inside `fabric.pe`, `fabric.instance` may target only named `fabric.pe`
    definitions, and cyclic instance graphs are not allowed.
  - See [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md).

See [spec-fabric-error.md](./spec-fabric-error.md) for error code definitions.

### Runtime Configuration Overrides

`fabric.instance` instantiates the hardware structure only. Runtime
configuration parameters are not copied from the referenced definition.

Rules:

- Any runtime configuration parameters present on the referenced definition are
  cleared when instantiating.
- The instance may specify runtime configuration parameters as attributes.
- The instance may only specify runtime configuration parameters that belong to
  the referenced operation.
- If the referenced definition contains non-default runtime configuration
  attributes, Loom emits a warning stating that they are cleared.

Clearing means the instance starts with the hardware's default reset values
(e.g., empty instruction memories, disabled routes, output tags at 0), unless
explicitly set on the instance.

### Semantics

`fabric.instance` is a hardware instantiation. Each instance represents a
separate stateful hardware component. The instance name is used for hardware
naming and debugging.

**Note on ADG Builder:** When using ADGBuilder to construct Fabric MLIR, all
runtime configuration attributes are left at default (empty) values. The
ADGBuilder exclusively describes hardware structure; runtime configuration is
populated by the compiler's place-and-route phase. See
[spec-adg.md](./spec-adg.md) for details.

## Operation: `fabric.yield`

A terminator for `fabric.module`, `fabric.pe`, and `fabric.temporal_pe` bodies.

### Syntax

```
fabric.yield %v0, %v1 : T0, T1
```

### Constraints

- `fabric.yield` must be the last operation in the region.
- Operand count and types must match the parent operation result types.

### Semantics

`fabric.yield` returns values from the operation body to the parent operation.
For `fabric.module`, the yielded values become the module outputs.

## Error Reporting and Codes

Fabric error classes and the global error code space are defined in
[spec-fabric-error.md](./spec-fabric-error.md). All references in this
document use the symbols defined there.

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-mapper.md](./spec-mapper.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-mem.md](./spec-fabric-mem.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-error.md](./spec-fabric-error.md)

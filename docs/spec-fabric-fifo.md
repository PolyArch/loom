# Fabric FIFO Specification

## Overview

A `fabric.fifo` is a single-input, single-output pipeline buffer. It introduces
pipeline delay into the routing network, breaking combinational loops and
providing backpressure support.

## Operation: `fabric.fifo`

### Forms

`fabric.fifo` supports two forms:

- **Named form**: defines a reusable FIFO module with a symbol name.
- **Inline form**: defines a local FIFO used directly in the surrounding region.

Both forms share the same semantics and constraints.

### Named Form Syntax

```mlir
fabric.fifo @buf [depth = 4] : (i32) -> (i32)
fabric.fifo @buf [depth = 4, bypassable] {bypassed = false} : (i32) -> (i32)
```

Named FIFOs can be instantiated via `fabric.instance`:

```mlir
%out = fabric.instance @buf(%in) {sym_name = "f0"} : (i32) -> i32
```

### Inline Form Syntax

```mlir
%out = fabric.fifo [depth = 4] %in : i32
%out = fabric.fifo [depth = 4, bypassable] {bypassed = false} %in : i32
%tagged_out = fabric.fifo [depth = 2] %tagged_in : !dataflow.tagged<i32, i4>
```

### Interface Types

- Single input, single output.
- Output type must be bit-width compatible with input type. For native types,
  only bit width must match (e.g., `i32` and `f32` are compatible). For
  `!dataflow.tagged` types, value bit width AND tag bit width must each match.
  Native-to-tagged mixing is never allowed.
- The type may be any native value type or `!dataflow.tagged`.

**Supported types:**
- Integer: `i1`, `i8`, `i16`, `i32`, `i64`
- Float: `f16`, `bf16`, `f32`, `f64`
- Special: `index`, `none`
- Tagged: `!dataflow.tagged<value_type, tag_type>`

FIFOs of same-width types are physically equivalent (e.g., `f32` vs `i32`
produce identical hardware).

## Hardware Parameters

| Attribute | Type | Range | Description |
|-----------|------|-------|-------------|
| `depth` | integer | >= 1 | Number of buffer slots |
| `bypassable` | flag (UnitAttr) | present/absent | Enables runtime bypass capability |

`depth` is a hardware parameter that determines the physical buffer size. It is
fixed for a given hardware instance and cannot be reconfigured at runtime.
`depth >= 1` is always required regardless of `bypassable`.

`bypassable` is a hardware parameter that, when present, adds bypass circuitry
to the FIFO. When bypassed at runtime, the input is directly connected to the
output, skipping the buffer. When `bypassable` is absent (the default), the
FIFO has no bypass capability.

## Runtime Configuration

| Attribute | Type | Condition | Default | Description |
|-----------|------|-----------|---------|-------------|
| `bypassed` | bool | only when `bypassable` is set | `false` | Bypass the FIFO buffer |

When `bypassable` is not set, `fabric.fifo` has no runtime configuration
parameters (`CONFIG_WIDTH = 0`).

When `bypassable` is set, `fabric.fifo` has a 1-bit runtime configuration
parameter `bypassed` (`CONFIG_WIDTH = 1`):

- `bypassed = false` (default): normal FIFO operation with buffering.
- `bypassed = true`: input directly connected to output, buffer skipped.

**Warning:** When `bypassed = true`, the FIFO becomes combinational (zero delay).
If the FIFO was the only sequential element breaking a combinational loop, bypassing
it creates a combinational loop at runtime. Avoiding this is the mapper's
responsibility, not a compile-time constraint. The FIFO is still classified as
**sequential** for compile-time combinational loop detection regardless of the
`bypassed` value.

## Depth Semantics

- **`depth = 0`**: NOT allowed. Raises `CPL_FIFO_DEPTH_ZERO`.
- **`depth = 1`**: Single-buffer. Upstream writes in cycle N, downstream reads
  in cycle N+1. Backpressure has 1-cycle latency. 50% throughput. Sufficient
  to break combinational loops.
- **`depth >= 2`**: Double-buffering enables full throughput. One slot accepts
  new data while the other delivers, so the pipeline never stalls due to the
  FIFO itself.

## Timing Model

`fabric.fifo` is a **sequential** element with minimum 1-cycle latency. This
property is what allows FIFOs to break combinational loops: a cycle containing
at least one FIFO is not a combinational loop because the FIFO introduces a
clock boundary.

## Backpressure Behavior

`fabric.fifo` participates in valid/ready handshaking:

- **Upstream**: accepts data when buffer is not full (ready = 1).
- **Downstream**: presents data when buffer is not empty (valid = 1).
- When full, upstream is backpressured (ready = 0).
- When empty, downstream sees no valid data (valid = 0).

## Constraints

| Error Code | Condition |
|------------|-----------|
| `CPL_FIFO_DEPTH_ZERO` | `depth` is 0 |
| `CPL_FIFO_TYPE_MISMATCH` | Input and output types are not bit-width compatible |
| `CPL_FIFO_INVALID_TYPE` | Type is not a native value type or `!dataflow.tagged` |
| `CPL_FIFO_BYPASSED_NOT_BYPASSABLE` | `bypassed` attribute present without `bypassable` |
| `CPL_FIFO_BYPASSED_MISSING` | `bypassable` is set but `bypassed` attribute is missing |

See [spec-fabric-error.md](./spec-fabric-error.md) for error code definitions.

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-error.md](./spec-fabric-error.md)
- [spec-adg-api.md](./spec-adg-api.md)

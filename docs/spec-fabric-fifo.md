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
```

Named FIFOs can be instantiated via `fabric.instance`:

```mlir
%out = fabric.instance @buf(%in) {sym_name = "f0"} : (i32) -> i32
```

### Inline Form Syntax

```mlir
%out = fabric.fifo [depth = 4] %in : i32
%tagged_out = fabric.fifo [depth = 2] %tagged_in : !dataflow.tagged<i32, i4>
```

### Interface Types

- Single input, single output.
- Output type must be identical to input type.
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

`depth` is a hardware parameter that determines the physical buffer size. It is
fixed for a given hardware instance and cannot be reconfigured at runtime.

## Runtime Configuration

`fabric.fifo` has no runtime configuration parameters.

## Depth Semantics

- **`depth = 0`**: NOT allowed. Raises `COMP_FIFO_DEPTH_ZERO`.
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
| `COMP_FIFO_DEPTH_ZERO` | `depth` is 0 |
| `COMP_FIFO_TYPE_MISMATCH` | Input and output types do not match |
| `COMP_FIFO_INVALID_TYPE` | Type is not a native value type or `!dataflow.tagged` |

See [spec-fabric-error.md](./spec-fabric-error.md) for error code definitions.

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-error.md](./spec-fabric-error.md)
- [spec-adg-api.md](./spec-adg-api.md)

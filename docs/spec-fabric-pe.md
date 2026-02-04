# Fabric Processing Element (PE) Specification

## Overview

A `fabric.pe` models a processing element that groups one or more operations
into a single hardware unit. It defines a small hardware subgraph with a
fixed interface, latency, and initiation interval.

A `fabric.pe` can be used directly as an inline operation or referenced via
`fabric.instance` when it has a symbol name.

## Operation: `fabric.pe`

### Forms

`fabric.pe` supports two forms:

- **Named form**: defines a reusable PE module with a symbol name.
- **Inline form**: defines a local PE used directly in the surrounding region.

Both forms share the same semantics and constraints.

### Named Form Syntax

```
fabric.pe @name(%arg0: T0, %arg1: T1) -> (R0)
    [latency = [0 : i16, 1 : i16, 2 : i16],
     interval = [1 : i16, 1 : i16, 1 : i16],
     output_tag = [0 : i4]] {
  // body
  fabric.yield %result : R0
}
```

### Inline Form Syntax

```
%out0, %out1 = fabric.pe %in0, %in1
    [latency = [1 : i16, 2 : i16, 3 : i16],
     interval = [1 : i16, 1 : i16, 1 : i16]]
    : (T0, T1) -> (R0, R1) {
  ^bb0(%a: T0, %b: T1):
    // body
    fabric.yield %v0, %v1 : R0, R1
}
```

### Interface Types

All ports must belong to the same category:

- **Native category**: every port is a native value type.
- **Tagged category**: every port is a `!dataflow.tagged<value_type, tag_type>`.

Within a category, individual port types may differ. This allows type
conversion PEs.

### Tag Visibility and Block Arguments

When the interface is tagged:

- The tag is not visible inside the `fabric.pe` body.
- The entry block arguments are the value types only.
- The body must yield value types only.
- The operation result types remain tagged; tags are reattached at the
  boundary using `output_tag`.

This can be viewed as implicit `fabric.del_tag` at inputs and implicit
`fabric.add_tag` at outputs.

### Attributes

#### `latency` (hardware parameter)

- Type: `ArrayAttr` of three `i16` values `[min, typical, max]`.
- Constraint: `min <= typical <= max`.
- Constraint: `min >= 0`.
- Semantics: cycles from consuming all inputs to producing outputs.

`latency = [0, 0, 0]` denotes a combinational PE.

#### `interval` (hardware parameter)

- Type: `ArrayAttr` of three `i16` values `[min, typical, max]`.
- Constraint: `min <= typical <= max`.
- Constraint: `min >= 1`.
- Semantics: cycles between consecutive input consumptions.

`interval = [1, 1, 1]` denotes a fully pipelined PE.

#### `output_tag` (runtime configuration parameter)

- Required only when the interface category is tagged.
- Type: `ArrayAttr` of integers, one per output port.
- Each element type must match the tag type of the corresponding output port.
- Default value for each element is `0`.
- May be reprogrammed at runtime.

The `output_tag` array provides one tag per output. Input tags are ignored and
are dropped at the boundary.

### Constraints

- All ports must be native types, or all ports must be tagged types.
- If the interface is native, `output_tag` must be absent.
- If the interface is tagged, `output_tag` is required, except inside
  `fabric.temporal_pe` where it is ignored.
- The body must contain at least one non-terminator operation.
- If any `dataflow` operation is present, the interface must be native and the
  body must be dataflow-only as defined below.

### Allowed Operations Inside `fabric.pe`

`fabric.pe` bodies may include operations from the following dialects:

- `arith`
- `math`
- LLVM arithmetic intrinsics
- `dataflow` (restricted to the four ops listed below)
- `fabric.pe` (nested)
- `fabric.instance` (to instantiate named PEs)
- Allowed `handshake` operations listed below

#### Handshake Allowlist

The following `handshake` operations are allowed inside `fabric.pe`:

- `handshake.cond_br`
- `handshake.mux`
- `handshake.load`
- `handshake.store`
- `handshake.constant`
- `handshake.sink`
- `handshake.fork`
- `handshake.join`

All other `handshake` operations are disallowed inside `fabric.pe`.

#### Dataflow Allowlist

The following `dataflow` operations are allowed inside `fabric.pe`:

- `dataflow.carry`
- `dataflow.invariant`
- `dataflow.stream`
- `dataflow.gate`

#### Homogeneous Consumption Rule

A `fabric.pe` body must be homogeneous with respect to input consumption and
output production behavior. There are two groups.

Full-consume/full-produce group: operations that consume all their inputs and
produce all outputs each firing. This group includes arithmetic operations and
`handshake.load`, `handshake.store`, `handshake.constant`, `handshake.sink`,
`handshake.fork`, and `handshake.join`.

Partial-consume/partial-produce group: operations that may consume only a
subset of inputs or produce only a subset of outputs per firing. This group
includes `handshake.cond_br` and `handshake.mux`.

A `fabric.pe` body must use operations from exactly one of these groups.
Mixing groups in a single `fabric.pe` is not allowed.

#### Load/Store Exclusivity Rule

If a `fabric.pe` body contains `handshake.load` or `handshake.store`, then the
body may contain only `handshake.load`, `handshake.store`, and the terminator
`fabric.yield`. No other operations are permitted in that case.

#### Dataflow Exclusivity Rule

If a `fabric.pe` body contains any `dataflow` operation, then the body may
contain only `dataflow` operations and the terminator `fabric.yield`. Mixing
`dataflow` with `arith`, `math`, LLVM intrinsics, `handshake`, or nested
`fabric.pe`/`fabric.instance` is not allowed.

Because the `dataflow` dialect does not support tagged types, a dataflow-only
`fabric.pe` must use the native interface category. Tagged interfaces are
invalid in this case.

### Prohibited Operations Inside `fabric.pe`

The following operations are not allowed inside `fabric.pe`:

- `fabric.switch`
- `fabric.temporal_pe`
- `fabric.temporal_sw`
- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`
- Any `handshake` operation not listed in the allowlist

If a software graph contains unsupported handshake operations, it cannot be
mapped into a `fabric.pe`.

### Example: Native PE with Type Conversion

```
fabric.pe @sitofp(%a: i32) -> (f32)
    [latency = [1 : i16, 1 : i16, 1 : i16],
     interval = [1 : i16, 1 : i16, 1 : i16]] {
  %v = arith.sitofp %a : i32 to f32
  fabric.yield %v : f32
}
```

### Example: Tagged Interface (Tag Hidden)

```
%out = fabric.pe %in0, %in1
    [latency = [0 : i16, 0 : i16, 0 : i16],
     interval = [1 : i16, 1 : i16, 1 : i16],
     output_tag = [3 : i4]]
    : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>)
      -> (!dataflow.tagged<i32, i4>) {
  ^bb0(%a: i32, %b: i32):
    %sum = arith.addi %a, %b : i32
    fabric.yield %sum : i32
}
```

## Interaction with `fabric.temporal_pe`

When a `fabric.pe` is used inside a `fabric.temporal_pe`:

- The PE still operates on value-only data in its body.
- Any `output_tag` on the PE is ignored.
- If `output_tag` is present, the compiler must emit a warning but must not
  treat it as an error.
- Output tags are taken from the `instruction_mem` of the enclosing
  `fabric.temporal_pe`.

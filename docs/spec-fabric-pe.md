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
Load/store PEs have additional interface rules described in
`Load/Store PE Semantics` below.

### Tag Visibility and Block Arguments

When the interface is tagged:

- The tag is not visible inside the `fabric.pe` body.
- The entry block arguments are the value types only.
- The body must yield value types only.
- The operation result types remain tagged; tags are reattached at the
  boundary using `output_tag`, except for load/store PEs in the
  tag-transparent hardware type where tags are preserved. See
  `Load/Store PE Semantics`.

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

- Allowed only when the interface category is tagged.
- Type: `ArrayAttr` of integers, one per output port.
- Each element type must match the tag type of the corresponding output port.
- Default value for each element is `0`.
- May be reprogrammed at runtime.

The `output_tag` array provides one tag per output. Input tags are ignored and
are dropped at the boundary.

Load/store PEs override the default `output_tag` rules. See
`Load/Store PE Semantics`.

#### `lqDepth` and `sqDepth` (hardware parameters)

These attributes are only valid for load/store PEs:

- `lqDepth` applies to a load PE.
- `sqDepth` applies to a store PE.

They define the depth of the internal tag-matching queues used by the
tag-transparent hardware type (no `output_tag`). See `Load/Store PE Semantics`.

### Constraints

- All ports must be native types, or all ports must be tagged types, except for
  the load/store PE special cases described below.
- If the interface is native, `output_tag` must be absent.
- If the interface is tagged, `output_tag` is required for non-load/store PEs.
  Load/store PEs follow their own rules below.
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

`dataflow.stream` supports the `step_op` and `stop_cond` attributes to encode
loop updates and termination conditions. See [spec-dataflow.md](./spec-dataflow.md).

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
body must contain exactly one of these operations and no other non-terminator
operations. The only allowed body shape is:

- A single `handshake.load` or a single `handshake.store`.
- The `fabric.yield` terminator.

Any other operation in the body, or a mix of load and store, is a compile-time
error (`COMP_PE_LOADSTORE_BODY`).

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

### Load/Store PE Semantics

Load/store PEs are hardware adapters between the fabric memory system and
compute graph. They only support a single `handshake.load` or a single
`handshake.store` in the body. Their behavior is defined here and in
[spec-fabric-mem.md](./spec-fabric-mem.md).

#### Port Roles

Load PE:

- Inputs: `addr_from_comp`, `ctrl`, `data_from_mem` (data returned from memory)
- Outputs: `addr_to_mem`, `data_to_comp` (data forwarded to compute)

Store PE:

- Inputs: `addr_from_comp`, `data_from_comp`, `ctrl`
- Outputs: `addr_to_mem`, `data_to_mem`

Tag consistency:

- For a load PE, `addr_from_comp`, `data_from_mem`, `addr_to_mem`, and
  `data_to_comp` must share the same taggedness and tag width.
- For a store PE, `addr_from_comp`, `data_from_comp`, `addr_to_mem`, and
  `data_to_mem` must share the same taggedness and tag width.
- Violations raise `COMP_PE_LOADSTORE_TAG_WIDTH`.

The `ctrl` input is a synchronization token. It is consumed and discarded after
the synchronization condition is met. Load/store PEs do not output a done token.
The done token is produced by `fabric.memory`/`fabric.extmemory` (`lddone` or
`stdone`) and must be wired directly into the control chain or sunk.

Synchronization rules:

- Load PE fires when `addr_from_comp` and `ctrl` are both ready.
- Store PE fires when `addr_from_comp`, `data_from_comp`, and `ctrl` are all ready.
- For load PEs, `data_from_mem` does not participate in the synchronization;
  it is forwarded independently.
- Tag matching applies only to Hardware Type B (tag-transparent mode). In
  Type B, inputs must have matching tags before the PE fires. Type A does not
  perform tag matching.

Tagged and native interfaces are not implicitly convertible. If a load/store PE
is tagged, it cannot connect directly to a single-port memory interface
(`ldCount == 1` or `stCount == 1`) because those ports are native. Use explicit
`fabric.del_tag` or `fabric.add_tag` boundaries to convert types.

#### Tag Modes

Load/store PEs are defined as two distinct hardware types. The hardware type is
fixed at build time and cannot be switched at runtime. Type A may be native or
tagged; Type B is tagged-only.

Hardware Type A: output-tag overwrite.

- If the interface is native, `output_tag` must be absent.
- If the interface is tagged, `output_tag` is required.
- The PE represents a single logical software edge.
- The `addr_from_comp` port and the data port (`data_from_mem` for load,
  `data_from_comp` for store) may be tagged or native.
- The `ctrl` port must be `none`.
- `lqDepth`/`sqDepth` must be absent.
- When tagged, output tags are overwritten with `output_tag`.
- When tagged, the `data_from_mem` tag is ignored; the output tag is always
  `output_tag`.
- When native, there is no tag overwrite.

Hardware Type B: tag-transparent.

- `output_tag` is absent.
- `addr_from_comp`, the data port (`data_from_mem` for load, `data_from_comp`
  for store), and `ctrl` ports must all be tagged.
- The `ctrl` value type is `i1` and the tag carries the logical port ID. The
  `i1` payload is a dummy constant `0` and must not drive logic. See
  [spec-dataflow.md](./spec-dataflow.md).
- Tag widths on `addr_from_comp`, the data port, and `ctrl` must match.
- `lqDepth` (load) or `sqDepth` (store) is required and must be >= 1.
- The PE synchronizes inputs only when tags match, then forwards tags
  unchanged to the outputs.
- The PE does not enforce tag equality between returned data and the matched
  request.

Invalid combinations of `output_tag`, tagged `ctrl`, and queue depth attributes
are compile-time errors (`COMP_PE_LOADSTORE_TAG_MODE`,
`COMP_PE_LOADSTORE_TAG_WIDTH`).

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
- The PE must use a native (non-tagged) interface. Tagged `fabric.pe` is not
  allowed inside `fabric.temporal_pe` (`COMP_TEMPORAL_PE_TAGGED_PE`).
- Load/store PEs are forbidden inside `fabric.temporal_pe`. The compiler must
  emit `COMP_TEMPORAL_PE_LOADSTORE`.

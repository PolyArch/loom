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

#### `constant_value` (runtime configuration parameter)

- Allowed only when the PE body contains exactly one `handshake.constant`.
- Type: matches the output value type of the `handshake.constant`.
- The value is stored in config_mem and may be reprogrammed at runtime.
- When the interface is tagged, `constant_value` is packed together with
  `output_tag` in config_mem (constant_value in lower bits, output_tag in
  upper bits).

This attribute is a direct consequence of the Constant Exclusivity Rule: if a
`fabric.pe` contains `handshake.constant`, it must be the only non-terminator
operation. Therefore, each PE can have at most one `constant_value`.

#### `lqDepth` and `sqDepth` (hardware parameters)

These attributes are only valid for load/store PEs:

- `lqDepth` applies to a load PE.
- `sqDepth` applies to a store PE.

They define the depth of the internal tag-matching queues used by the
tag-transparent hardware type (no `output_tag`). See `Load/Store PE Semantics`.

### Constraints

- All ports must be native types, or all ports must be tagged types, except for
  the load/store PE special cases described below. Violations raise
  `COMP_PE_MIXED_INTERFACE`.
- If the interface is native, `output_tag` must be absent. Violations raise
  `COMP_PE_OUTPUT_TAG_NATIVE`.
- If the interface is tagged, `output_tag` is required for non-load/store PEs.
  Load/store PEs follow their own rules below. Violations raise
  `COMP_PE_OUTPUT_TAG_MISSING`.
- The body must contain at least one non-terminator operation. Violations raise
  `COMP_PE_EMPTY_BODY`.
- If any `dataflow` operation is present, the interface must be native and the
  body must be dataflow-only as defined below. Violations raise
  `COMP_PE_DATAFLOW_BODY`.

See [spec-fabric-error.md](./spec-fabric-error.md) for error code definitions.

### Allowed Operations Inside `fabric.pe`

See [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md) for the complete list of
allowed operations. This includes operations from:

- `arith` (30 operations)
- `math` (7 operations)
- `dataflow` (4 operations, with exclusivity constraint)
- `handshake` (8 operations, with exclusivity constraints)
- `fabric.pe` (nested)
- `fabric.instance` (to instantiate named PEs)

#### Body Constraints and Exclusivity Rules

All body constraints (homogeneous consumption rule, load/store exclusivity,
dataflow exclusivity, constant exclusivity, instance-only prohibition, and
prohibited operations) are defined authoritatively in
[spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md). Refer to that document
for the complete specification.

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
- Tag matching applies only to TagTransparent hardware type (tag-transparent mode). In
  TagTransparent, inputs must have matching tags before the PE fires. TagOverwrite does not
  perform tag matching.

Tagged and native interfaces are not implicitly convertible. If a load/store PE
is tagged, it cannot connect directly to a single-port memory interface
(`ldCount == 1` or `stCount == 1`) because those ports are native. Use explicit
`fabric.del_tag` or `fabric.add_tag` boundaries to convert types.

#### Hardware Types

Load/store PEs are defined as two distinct hardware types. The hardware type is
fixed at build time and cannot be switched at runtime.

**TagOverwrite** (output-tag overwrite):

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

**TagTransparent** (tag-transparent):

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

See [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) for the
authoritative constraints on `fabric.pe` usage within `fabric.temporal_pe`.
Key restrictions: the PE must use a native interface
(`COMP_TEMPORAL_PE_TAGGED_PE`) and load/store PEs are forbidden
(`COMP_TEMPORAL_PE_LOADSTORE`).

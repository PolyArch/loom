# Fabric Switch Specification

## Overview

A `fabric.switch` is a configurable routing switch. It routes one input to each
output based on a fixed physical connectivity table and a runtime route table.

## Operation: `fabric.switch`

### Forms

`fabric.switch` supports two forms:

- **Named form**: defines a reusable switch module with a symbol name.
- **Inline form**: defines a local switch used directly in the surrounding
  region.

Both forms share the same semantics and constraints.

### Named Form Syntax

```mlir
fabric.switch @sw4x4
  [connectivity_table = [ ... ]]
  {route_table = [ ... ]}
  : (T, T, T, T) -> (T, T, T, T)
```

Named switches can be instantiated via `fabric.instance`:

```mlir
%o0, %o1, %o2, %o3 = fabric.instance @sw4x4(%i0, %i1, %i2, %i3)
  : (T, T, T, T) -> (T, T, T, T)
```

### Inline Form Syntax

```mlir
%out0, %out1 = fabric.switch
  [connectivity_table = [...]]
  {route_table = [...]}
  %in0, %in1, %in2 : T -> T, T
```

### Interface Types

- All inputs and outputs must share the same type.
- The type may be a native value type or `!dataflow.tagged`.
- If the type is tagged, the tag is forwarded unchanged.

### Attributes

#### `connectivity_table` (hardware parameter)

- Type: `DenseI8ArrayAttr`.
- Values: `0` or `1`.
- Shape: `num_outputs * num_inputs` in row-major order.
- Indexing convention: `connectivity_table[output][input]`.
- Entry `1` means a physical connection exists from input to output.
- Entry `0` means no physical connection exists.

#### `route_table` (runtime configuration parameter)

- Type: `DenseI8ArrayAttr`.
- Values: `0` or `1`.
- Length: equal to the number of `1` entries in `connectivity_table`.
- Ordering: entries correspond to `1` positions in `connectivity_table`,
  scanned in row-major order by output then input.
- Config bit width: `K` bits, where `K = popcount(connectivity_table)`.

A `1` in `route_table` enables a connected path. A `0` disables it.
For the formal `config_mem` definition and packing rules, see
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

### Defaults

If an attribute is omitted, the following defaults apply:

- `connectivity_table`: all `1` (full crossbar connectivity).
- `route_table`: all `0` (no routes enabled).

### Constraints

- The number of inputs and outputs must each be less than or equal to 32.
- `connectivity_table` length must equal `num_outputs * num_inputs`.
- `route_table` length must equal the number of `1` entries in
  `connectivity_table`.
- Each output row of `connectivity_table` must have at least one `1`.
- Each input column of `connectivity_table` must have at least one `1`.
- Each output may select at most one routed input.

Violations of hardware-parameter constraints are compile-time errors:
`CPL_SWITCH_PORT_LIMIT`, `CPL_SWITCH_TABLE_SHAPE`, `CPL_SWITCH_ROW_EMPTY`,
`CPL_SWITCH_COL_EMPTY`, and `CPL_SWITCH_ROUTE_LEN_MISMATCH`. Violations of
runtime routing constraints are configuration errors:
`CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT` (multiple inputs route to the same
output). See [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`fabric.switch` models a physical routing resource:

- `connectivity_table` defines the physical wires.
- `route_table` enables a subset of those wires at runtime.

When an output is connected to exactly one routed input, the output forwards
that input. If an output has no routed input, the output produces no token.

**Broadcast**: One input can route to multiple outputs (the route_table column
for that input may have multiple 1s across different output rows). When
broadcasting, the input's ready signal is the AND of all targeted outputs'
ready signals (atomic delivery).

One output still receives from at most one input
(`CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT` enforced).

**Broadcast valid/ready**: `out_valid[j]` depends only on `in_valid[source]`,
with no dependency on any `out_ready` signal. This avoids combinational loops
through downstream consumers' ready paths.

- `out_valid[j] = in_valid[source]`.
- `in_ready[i] = AND(out_ready[k])` for all outputs `k` targeted by input `i`.

Atomic broadcast is guaranteed by `in_ready`: the source only advances when
ALL broadcast targets have consumed the data (AND of all targeted readys).

### Timing Model

`fabric.switch` uses a combinational datapath with latched error reporting:

- Data-path routing is combinational (zero-cycle forwarding once
  valid/ready allows transfer).
- Runtime routing-constraint checks are combinational, but the first detected
  error is captured into `error_valid`/`error_code` and held until reset.
- `error_valid` is sticky (once set, it remains asserted until reset), and
  later errors do not overwrite the first captured error code.
- When multiple error conditions are true in the same cycle, the error with the
  numerically smallest error code is captured. See
  [spec-fabric-error.md](./spec-fabric-error.md) for the cross-module
  precedence rule.

### Backpressure Behavior

The switch uses standard valid/ready handshaking. Backpressure propagates from
outputs to inputs:

- An input is blocked (backpressured) when any of its targeted outputs is not
  ready (broadcast: AND of all targeted output readys).
- Non-broadcasting paths operate independently; blocking one path does not
  affect other paths.

### Unrouted Input Error

If an input that has physical connectivity (a `1` in `connectivity_table`) but
no enabled route (no `1` in the corresponding `route_table` positions) receives
a valid token, the switch raises a runtime error (`RT_SWITCH_UNROUTED_INPUT`).
This prevents silent data loss from misconfigured routes.

See [spec-fabric-error.md](./spec-fabric-error.md).

### Example

A 3-input, 2-output switch with partial connectivity:

```
%o0, %o1 = fabric.switch
  [connectivity_table = [0, 1, 1, 1, 1, 0]]
  {route_table = [1, 0, 1, 0]}
  %i0, %i1, %i2 : i32 -> i32, i32
```

Interpretation:

- `connectivity_table` entries are ordered as:
  - Output 0: inputs 0, 1, 2
  - Output 1: inputs 0, 1, 2
- Physical connections are:
  - Output 0: input 1 and input 2
  - Output 1: input 0 and input 1
- `route_table` enables:
  - Output 0 <- input 1
  - Output 1 <- input 0

See [spec-fabric.md](./spec-fabric.md) Operation Syntax Conventions for the
canonical `[hw_params] {runtime_config}` bracket convention.

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-error.md](./spec-fabric-error.md)

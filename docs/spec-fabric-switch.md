# Fabric Switch Specification

## Overview

A `fabric.switch` is a configurable routing switch. It routes one input to each
output based on a fixed physical connectivity table and a runtime route table.

## Operation: `fabric.switch`

### Syntax

```
%out0, %out1 = fabric.switch [connectivity_table = [...], route_table = [...]]
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
- Entry `1` means a physical connection exists from input to output.
- Entry `0` means no physical connection exists.

#### `route_table` (runtime configuration parameter)

- Type: `DenseI8ArrayAttr`.
- Values: `0` or `1`.
- Length: equal to the number of `1` entries in `connectivity_table`.
- Ordering: entries correspond to `1` positions in `connectivity_table`,
  scanned in row-major order by output then input.

A `1` in `route_table` enables a connected path. A `0` disables it.

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
- Each input may route to at most one output.

Violations of hardware-parameter constraints are compile-time errors:
`COMP_SWITCH_PORT_LIMIT`, `COMP_SWITCH_TABLE_SHAPE`, `COMP_SWITCH_ROW_EMPTY`,
`COMP_SWITCH_COL_EMPTY`, and `COMP_SWITCH_ROUTE_LEN_MISMATCH`. Violations of
runtime routing constraints are configuration errors: `CFG_SWITCH_ROUTE_MULTI_OUT`
and `CFG_SWITCH_ROUTE_MULTI_IN`. See [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`fabric.switch` models a physical routing resource:

- `connectivity_table` defines the physical wires.
- `route_table` enables a subset of those wires at runtime.

When an output is connected to exactly one routed input, the output forwards
that input. If an output has no routed input, the output produces no token.

### Backpressure Behavior

The switch uses standard valid/ready handshaking. Backpressure propagates from
outputs to inputs:

- An input is blocked (backpressured) when its destination output is not ready.
- Each input-output pair operates independently; blocking one path does not
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
  [connectivity_table = [0, 1, 1, 1, 1, 0],
   route_table = [1, 0, 1, 0]]
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

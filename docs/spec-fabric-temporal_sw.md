# Fabric Temporal Switch Specification

## Overview

A `fabric.temporal_sw` is a tag-aware routing switch. It selects a routing
configuration based on the tag of the incoming token. The tag is used only for
selection and is forwarded unchanged to the output.

## Operation: `fabric.temporal_sw`

### Syntax

```
%out0, %out1 = fabric.temporal_sw
  [num_route_table = 4, connectivity_table = [...]]
  {route_table = [ ... ]}
  %in0, %in1, %in2 : !dataflow.tagged<T, iN> -> !dataflow.tagged<T, iN>, !dataflow.tagged<T, iN>
```

### Interface Types

- All inputs and outputs must be `!dataflow.tagged`.
- All ports must use the same tagged type.
- Tag width must be in the range `i1` to `i16`.

### Attributes

#### `num_route_table` (hardware parameter)

- Unsigned integer.
- Maximum number of routing table slots supported by hardware.
- Must be greater than 0.

#### `connectivity_table` (hardware parameter)

- Type: `DenseI8ArrayAttr`.
- Values: `0` or `1`.
- Shape: `num_outputs * num_inputs` in row-major order.
- Entry `1` means a physical connection exists from input to output.
- Entry `0` means no physical connection exists.

#### `route_table` (runtime configuration parameter)

- ArrayAttr of strings, either human-readable or hex format.
- Length must be less than or equal to `num_route_table`.
- All entries must use the same format.

### Defaults

If an attribute is omitted, the following defaults apply:

- `connectivity_table`: all `1` (full crossbar connectivity).
- `route_table`: all slots invalid. This is equivalent to all `invalid`
  entries in human-readable format or all `0x0` entries in machine format.

### Constraints

- The number of inputs and outputs must each be less than or equal to 32.
- `connectivity_table` length must equal `num_outputs * num_inputs`.
- Each output row of `connectivity_table` must have at least one `1`.
- Each input column of `connectivity_table` must have at least one `1`.
- Route entries must only enable positions that are connected in
  `connectivity_table`.
- Each output may select at most one routed input per slot.
- Each input may route to at most one output per slot.
- Each slot must have a unique tag value.

Hardware-parameter violations are compile-time errors: `COMP_TEMPORAL_SW_PORT_LIMIT`,
`COMP_TEMPORAL_SW_TABLE_SHAPE`, `COMP_TEMPORAL_SW_ROW_EMPTY`,
`COMP_TEMPORAL_SW_COL_EMPTY`, `COMP_TEMPORAL_SW_NUM_ROUTE_TABLE`,
`COMP_TEMPORAL_SW_TOO_MANY_SLOTS`, and `COMP_TEMPORAL_SW_ROUTE_ILLEGAL`.
Duplicate tags in `route_table` are configuration errors: `CFG_TEMPORAL_SW_DUP_TAG`.
If no slot matches an input tag at runtime, the temporal switch raises
`RT_TEMPORAL_SW_NO_MATCH`. See [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`fabric.temporal_sw` contains multiple routing tables (route_table slots). Each
slot is selected by tag matching:

- The input tag is compared against the tag of each valid slot.
- Exactly one slot must match the input tag.
- The selected slot determines the routed connections.
- The tag is forwarded unchanged to the output.

If no slot matches, the temporal switch raises a runtime error
(`RT_TEMPORAL_SW_NO_MATCH`). If multiple slots match, the configuration is
invalid due to duplicate tags (`CFG_TEMPORAL_SW_DUP_TAG`). The hardware emits
an error-valid signal and an error code that is propagated to the top level.
See [spec-fabric-error.md](./spec-fabric-error.md).

The temporal switch is input-driven. For a selected route, the output forwards
the chosen input as soon as it is valid and ready. The switch does not wait for
unrelated inputs.

If multiple inputs with different tags target the same output (through their
respective route_table slots), the output uses round-robin arbitration starting
from lower port index.

### Unrouted Input Error

If an input receives a valid token but the matched route_table slot does not
enable a route for that input, the temporal switch raises a runtime error
(`RT_TEMPORAL_SW_UNROUTED_INPUT`). This applies when the tag matches a slot
but the slot's routes do not include that input. This prevents silent data
loss from misconfigured route tables.

See [spec-fabric-error.md](./spec-fabric-error.md).

## Route Table Format

The route table format is identical to the temporal PE instruction sparse
format rules and supports human-readable and hex formats.

### Sparse Format Rules

For human-readable entries:

- Slot indices must be strictly ascending.
- Implicit holes are allowed only when there are no explicit `invalid` entries.
- If any explicit `invalid` entry is present, all holes must be explicit.
- Trailing invalid entries may be omitted.

For machine format entries:

- The array is dense by index.
- Only trailing `0x0` entries may be omitted.

### Human-Readable Format

```
route_table[slot]: when(tag=TAG) O[out]<-I[in], O[out]<-I[in], ...
route_table[slot]: invalid
```

Each route entry specifies one or more input-to-output connections. The order
of listed connections does not affect the slot encoding.

### Machine Format (Hex)

Each entry is a hexadecimal string:

```
0x<hex_value>
```

Bit layout is from LSB to MSB:

```
| valid | tag | routes |
```

ASCII diagram:

```
+--------------------------------------------------------------+
|                ROUTE TABLE SLOT (LSB -> MSB)                 |
+--------+---------+-------------------------------+-----------+
| valid  | tag[M]  | routes[K] (LSB-first)         |    MSB    |
+--------+---------+-------------------------------+-----------+
```

Definitions:

- `valid`: 1 bit. `0` means invalid slot.
- `tag`: `M` bits, where `M` is the tag width.
- `routes`: `K` bits, where `K` is the number of connected positions in
  `connectivity_table`.

`routes` is encoded in row-major order by output then input, considering only
positions where `connectivity_table` is `1`.

`routes` bits are stored LSB-first: position 0 corresponds to the LSB of the
`routes` field, and the last position corresponds to the MSB. This matches the
LSB-to-MSB convention used elsewhere in the fabric specification.

Slot width in bits:

```
slot_width = 1 + M + K
```

Where:

- `M` = tag bit width
- `K` = number of connected positions in `connectivity_table`

Complete example bitmap (LSB -> MSB):

Parameters:
- `M = 4`
- `K = 3` (three connected positions)

Example slot:
- `valid = 1`
- `tag = 5` (`0101`)
- `routes = 101` (positions 0 and 2 enabled, position 1 disabled; LSB-first)

Bitmap (fields separated by `|`):

```
1 | 0101 | 101
```

Hex encoding (8 bits, LSB -> MSB):

```
0xAB
```

### Format Selection Guidance

- Use the human-readable format for documentation and inspection.
- Use the machine (hex) format for hardware configuration files and simulation.

### Example

```
%o0, %o1 = fabric.temporal_sw
  [num_route_table = 4, connectivity_table = [1, 1, 0, 0, 1, 1]]
  {route_table = [
    "route_table[0]: when(tag=0) O[0]<-I[0]",
    "route_table[1]: when(tag=1) O[0]<-I[1], O[1]<-I[2]",
    "route_table[2]: when(tag=5) O[1]<-I[1]",
    "route_table[3]: invalid"
  ]}
  %i0, %i1, %i2 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
               -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
```

# Fabric Dialect Specification

## Overview

The fabric dialect is the hardware IR for Loom. It models a full accelerator as
interconnected hardware modules with explicit streaming and memory interfaces.
The dialect is designed to be a mapping target for software dataflow graphs.

This document defines the fabric dialect at a top level and provides full
specifications for `fabric.module`, `fabric.instance`, and `fabric.yield`. Other
operations are specified in dedicated documents referenced below.

## Operation Summary

| Category | Operations | Details |
|----------|------------|---------|
| Top level | `fabric.module` | This document |
| Instantiation | `fabric.instance` | This document |
| Terminator | `fabric.yield` | This document |
| Processing | `fabric.pe`, `fabric.temporal_pe` | [spec-fabric-pe.md](./spec-fabric-pe.md), [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) |
| Routing | `fabric.switch`, `fabric.temporal_sw` | [spec-fabric-switch.md](./spec-fabric-switch.md), [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md) |
| Tag boundary | `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag` | [spec-fabric-tag.md](./spec-fabric-tag.md) |
| Memory | `fabric.mem` | [spec-fabric-mem.md](./spec-fabric-mem.md) |

## Type Conventions

The fabric dialect uses the dataflow tagged type for temporal routing and
streaming:

- `!dataflow.tagged<value_type, tag_type>`

See [spec-dataflow.md](./spec-dataflow.md) for the tagged type definition.

Within fabric operations, the term **native value type** refers to scalar
integer, scalar floating-point, or `index` types. Vector, tensor, memref,
complex, and opaque types are not considered native value types.

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
- `fabric.mem`
- `fabric.instance`
- `fabric.yield`

Operations from `arith`, `math`, `handshake`, `dataflow`, and LLVM must be
contained inside `fabric.pe` and are not allowed directly at module level.

The body must end with `fabric.yield` and the yielded values must match the
module result types and ordering.

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
  - `fabric.mem`
- `fabric.add_tag`, `fabric.map_tag`, and `fabric.del_tag` cannot be
  instantiated. They must be used inline.
- Operand count and types must match the referenced module signature.
- Result count and types must match the referenced module signature.

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

Fabric errors are classified into three classes:

- **COMP_**: compile-time errors raised by the Loom compiler. These are not
  hardware error codes.
- **CFG_**: runtime configuration errors caused by invalid runtime parameters
  (configuration registers or tables). These are reported by hardware.
- **RT_**: runtime execution errors not caused by configuration. These are
  reported by hardware.

CFG_ errors are detected after writing runtime configuration registers or
tables, before or during execution, and do not require dataflow execution to
surface.

Only CFG_ and RT_ use hardware error codes. The code space below is reserved.
Implementations may extend the list but must not reuse reserved values.
RT_ error codes start at 256 and increase sequentially.

### COMP_ (Compile-Time Errors, No Hardware Code)

| Name | Condition |
|------|-----------|
| COMP_SWITCH_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| COMP_SWITCH_ROW_EMPTY | A connectivity row has no `1` entries |
| COMP_SWITCH_COL_EMPTY | A connectivity column has no `1` entries |
| COMP_SWITCH_PORT_LIMIT | `fabric.switch` has more than 32 inputs or outputs |
| COMP_SWITCH_ROUTE_LEN_MISMATCH | `route_table` length does not match connected positions |
| COMP_TEMPORAL_SW_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| COMP_TEMPORAL_SW_ROW_EMPTY | A connectivity row has no `1` entries |
| COMP_TEMPORAL_SW_COL_EMPTY | A connectivity column has no `1` entries |
| COMP_TEMPORAL_SW_PORT_LIMIT | `fabric.temporal_sw` has more than 32 inputs or outputs |
| COMP_TEMPORAL_SW_NUM_ROUTE_TABLE | `num_route_table` is less than 1 |
| COMP_TEMPORAL_SW_TOO_MANY_SLOTS | `route_table` entries exceed `num_route_table` |
| COMP_TEMPORAL_SW_ROUTE_ILLEGAL | A route entry targets a disconnected position |
| COMP_TEMPORAL_PE_NUM_INSTRUCTION | `num_instruction` is less than 1 |
| COMP_TEMPORAL_PE_NUM_INSTANCE | `num_instance` is 0 when `num_register > 0`, or nonzero when `num_register = 0` |
| COMP_TEMPORAL_PE_REG_DISABLED | An instruction uses `reg(idx)` when `num_register = 0` |
| COMP_MAP_TAG_TABLE_SIZE | `table_size` is out of range [1, 256] |
| COMP_MAP_TAG_TABLE_LENGTH | `table` length does not equal `table_size` |

### CFG_ (Runtime Configuration Errors, Hardware Code)

| Code | Name | Condition |
|------|------|-----------|
| 0 | OK | No error |
| 1 | CFG_SWITCH_ROUTE_MULTI_OUT | A single output routes multiple inputs |
| 2 | CFG_SWITCH_ROUTE_MULTI_IN | A single input routes multiple outputs |
| 3 | CFG_TEMPORAL_SW_DUP_TAG | Duplicate tags in `route_table` slots |
| 4 | CFG_TEMPORAL_PE_DUP_TAG | Duplicate tags in `instruction_mem` |
| 5 | CFG_TEMPORAL_PE_ILLEGAL_REG | Register index encodes a value >= `num_register` |
| 6 | CFG_TEMPORAL_PE_REG_TAG_NONZERO | `res_tag != 0` when writing a register |
| 7 | CFG_MAP_TAG_DUP_TAG | `map_tag` has multiple valid entries with the same `src_tag` |

### RT_ (Runtime Execution Errors, Hardware Code)

| Code | Name | Condition |
|------|------|-----------|
| 256 | RT_TEMPORAL_PE_NO_MATCH | Input tag matches no instruction |
| 257 | RT_TEMPORAL_SW_NO_MATCH | Input tag matches no route table slot |
| 258 | RT_MAP_TAG_NO_MATCH | `map_tag` finds no valid entry |

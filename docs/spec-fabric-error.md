# Fabric Error Codes Specification

## Overview

This document defines the global error code space for Fabric hardware.

Authoritative source files for error code definitions:
- **COMP_ codes (C++)**: `include/loom/Hardware/Common/FabricError.h`
- **CFG_/RT_ codes (SV)**: `lib/loom/Hardware/SystemVerilog/Common/fabric_error.svh`

Other documents reference these symbols by name.

Fabric errors are classified into three classes:

- **COMP_**: compile-time errors raised by the Loom compiler. These have no
  hardware error code.
- **CFG_**: runtime configuration errors caused by invalid runtime parameters
  (configuration registers or tables). These are reported by hardware.
- **RT_**: runtime execution errors not caused by configuration. These are
  reported by hardware.

CFG_ errors are detected after writing runtime configuration registers or
tables, before or during execution, and do not require dataflow execution to
surface.

CFG_ error codes currently use the range 0-10. Codes 11-255 are reserved for
future use.

RT_ error codes start at 256 and increase sequentially.

## COMP_ (Compile-Time Errors, No Hardware Code)

| Symbol | Condition |
|--------|-----------|
| COMP_SWITCH_PORT_ZERO | `fabric.switch` has zero inputs or outputs |
| COMP_SWITCH_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| COMP_SWITCH_ROW_EMPTY | A connectivity row has no `1` entries |
| COMP_SWITCH_COL_EMPTY | A connectivity column has no `1` entries |
| COMP_SWITCH_PORT_LIMIT | `fabric.switch` has more than 32 inputs or outputs |
| COMP_SWITCH_ROUTE_LEN_MISMATCH | `route_table` length does not match connected positions |
| COMP_TEMPORAL_SW_PORT_ZERO | `fabric.temporal_sw` has zero inputs or outputs |
| COMP_TEMPORAL_SW_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| COMP_TEMPORAL_SW_ROW_EMPTY | A connectivity row has no `1` entries |
| COMP_TEMPORAL_SW_COL_EMPTY | A connectivity column has no `1` entries |
| COMP_TEMPORAL_SW_PORT_LIMIT | `fabric.temporal_sw` has more than 32 inputs or outputs |
| COMP_TEMPORAL_SW_INTERFACE_NOT_TAGGED | `fabric.temporal_sw` interface type is not tagged |
| COMP_TEMPORAL_SW_NUM_ROUTE_TABLE | `num_route_table` is less than 1 |
| COMP_TEMPORAL_SW_TOO_MANY_SLOTS | `route_table` entries exceed `num_route_table` |
| COMP_TEMPORAL_SW_ROUTE_ILLEGAL | A route entry targets a disconnected position |
| COMP_TEMPORAL_SW_MIXED_FORMAT | `route_table` mixes human-readable and hex entries |
| COMP_TEMPORAL_SW_SLOT_ORDER | Human-readable `route_table` slot indices are not strictly ascending |
| COMP_TEMPORAL_SW_IMPLICIT_HOLE | `route_table` has implicit holes when explicit `invalid` entries exist |
| COMP_TEMPORAL_PE_INTERFACE_NOT_TAGGED | `fabric.temporal_pe` interface type is not tagged |
| COMP_TEMPORAL_PE_NUM_INSTRUCTION | `num_instruction` is less than 1 |
| COMP_TEMPORAL_PE_REG_FIFO_DEPTH | `reg_fifo_depth` is 0 when `num_register > 0`, or nonzero when `num_register = 0` |
| COMP_TEMPORAL_PE_EMPTY_BODY | A `fabric.temporal_pe` contains no FU definitions in its body |
| COMP_TEMPORAL_PE_FU_INVALID | An FU definition references an invalid or unresolvable PE name |
| COMP_TEMPORAL_PE_TAGGED_FU | An FU definition inside `fabric.temporal_pe` has tagged input or output ports |
| COMP_TEMPORAL_PE_FU_ARITY | An FU port count does not match the `fabric.temporal_pe` interface arity |
| COMP_TEMPORAL_PE_REG_DISABLED | An instruction uses `reg(idx)` when `num_register = 0` |
| COMP_TEMPORAL_PE_SRC_MISMATCH | A human-readable source uses `in(j)` where `j` is not the operand position |
| COMP_TEMPORAL_PE_TOO_MANY_SLOTS | `instruction_mem` entries exceed `num_instruction` |
| COMP_TEMPORAL_PE_MIXED_FORMAT | `instruction_mem` mixes human-readable and hex entries |
| COMP_TEMPORAL_PE_SLOT_ORDER | Human-readable `instruction_mem` slot indices are not strictly ascending |
| COMP_TEMPORAL_PE_IMPLICIT_HOLE | `instruction_mem` has implicit holes when explicit `invalid` entries exist |
| COMP_TEMPORAL_PE_DEST_COUNT | Destination count in a human-readable instruction does not equal `num_outputs` |
| COMP_TEMPORAL_PE_SRC_COUNT | Source count in a human-readable instruction does not equal `num_inputs` |
| COMP_TEMPORAL_PE_TAG_WIDTH | `K != num_bits(T)` or tag-width metadata is inconsistent for interface `!dataflow.tagged<T, iJ>` |
| COMP_TEMPORAL_PE_TAGGED_PE | A `fabric.temporal_pe` contains a tagged `fabric.pe` |
| COMP_TEMPORAL_PE_LOADSTORE | A `fabric.temporal_pe` contains a load/store PE |
| COMP_TEMPORAL_PE_FU_WIDTH | An FU port data width exceeds the `fabric.temporal_pe` interface value type width |
| COMP_MAP_TAG_TABLE_SIZE | `table_size` is out of range [1, 256] |
| COMP_MAP_TAG_TABLE_LENGTH | `table` length does not equal `table_size` |
| COMP_MAP_TAG_VALUE_TYPE_MISMATCH | `fabric.map_tag` output value type does not match input value type |
| COMP_ADD_TAG_VALUE_TYPE_MISMATCH | `fabric.add_tag` result value type does not match input type |
| COMP_ADD_TAG_VALUE_OVERFLOW | `fabric.add_tag` configured `tag` value exceeds the representable range of the output tag type |
| COMP_DEL_TAG_VALUE_TYPE_MISMATCH | `fabric.del_tag` output type does not match input tagged value type |
| COMP_TAG_WIDTH_RANGE | Tag type width is outside the allowed range defined by `!dataflow.tagged` (`i1` to `i16`) |
| COMP_MEMORY_PORTS_EMPTY | `ldCount == 0` and `stCount == 0` |
| COMP_MEMORY_LSQ_WITHOUT_STORE | `lsqDepth != 0` when `stCount == 0` |
| COMP_MEMORY_LSQ_MIN | `lsqDepth < 1` when `stCount > 0` |
| COMP_MEMORY_ADDR_TYPE | Address port is not `index` or `!dataflow.tagged<index, iK>` |
| COMP_MEMORY_DATA_TYPE | Data port element type does not match memref element type |
| COMP_MEMORY_TAG_REQUIRED | `ldCount > 1` or `stCount > 1` but ports are not tagged |
| COMP_MEMORY_TAG_FOR_SINGLE | Tagged ports used when `ldCount == 1` or `stCount == 1` |
| COMP_MEMORY_TAG_WIDTH | Tag width is smaller than `log2Ceil(count)` or mismatched across ports |
| COMP_MEMORY_STATIC_REQUIRED | `fabric.memory` uses a dynamic memref type |
| COMP_MEMORY_PRIVATE_OUTPUT | `fabric.module` yields a memref not produced by `fabric.memory` with `is_private = false` |
| COMP_MEMORY_EXTMEM_BINDING | `fabric.extmemory` memref operand is not a `fabric.module` memref input |
| COMP_MEMORY_EXTMEM_PRIVATE | `is_private` is supplied on `fabric.extmemory` |
| COMP_FABRIC_TYPE_MISMATCH | A connection inside `fabric.module` uses mismatched types or bit widths without an explicit conversion |
| COMP_MODULE_PORT_ORDER | `fabric.module` ports are not in the required order: memref*, native*, tagged* |
| COMP_MODULE_EMPTY_BODY | `fabric.module` body contains no operations other than the terminator |
| COMP_MODULE_MISSING_YIELD | `fabric.module` body does not end with `fabric.yield` |
| COMP_PE_LOADSTORE_BODY | A load/store PE does not contain exactly one `handshake.load` or `handshake.store` |
| COMP_PE_LOADSTORE_TAG_MODE | Invalid combination of `output_tag`, tagged ctrl, and queue depth attributes |
| COMP_PE_LOADSTORE_TAG_WIDTH | Tag widths do not match across addr/data/ctrl ports |
| COMP_PE_CONSTANT_BODY | A constant PE contains operations other than a single `handshake.constant` |
| COMP_PE_INSTANCE_ONLY_BODY | A `fabric.pe` body contains only a single `fabric.instance` with no other operations |
| COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE | `operand_buffer_size` is set when `enable_share_operand_buffer = false` |
| COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING | `operand_buffer_size` is missing when `enable_share_operand_buffer = true` |
| COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE | `operand_buffer_size` is out of range [1, 8192] |
| COMP_PE_INSTANCE_ILLEGAL_TARGET | Inside `fabric.pe`, `fabric.instance` targets a non-PE operation (only named `fabric.pe` targets are legal) |
| COMP_INSTANCE_OPERAND_MISMATCH | `fabric.instance` operand count or types do not match the referenced module signature |
| COMP_INSTANCE_RESULT_MISMATCH | `fabric.instance` result count or types do not match the referenced module signature |
| COMP_INSTANCE_UNRESOLVED | `fabric.instance` references a symbol that does not exist |
| COMP_INSTANCE_CYCLIC_REFERENCE | `fabric.instance` forms a cyclic reference in a scope that requires acyclic instantiation (for example, inside `fabric.pe`) |
| COMP_PE_EMPTY_BODY | A `fabric.pe` body contains no non-terminator operations |
| COMP_PE_MIXED_INTERFACE | A `fabric.pe` has mixed native and tagged ports |
| COMP_PE_TAGGED_INTERFACE_NATIVE_PORTS | A `fabric.pe` with tagged interface category has native (untagged) ports |
| COMP_PE_NATIVE_INTERFACE_TAGGED_PORTS | A `fabric.pe` with native interface category has tagged ports |
| COMP_PE_DATAFLOW_BODY | A `fabric.pe` dataflow body violates dataflow exclusivity: mixed dataflow/non-dataflow ops, multiple dataflow ops, or any `fabric.instance` nesting |
| COMP_PE_MIXED_CONSUMPTION | A `fabric.pe` body mixes full-consume and partial-consume operations |
| COMP_PE_OUTPUT_TAG_NATIVE | A native `fabric.pe` has `output_tag` attribute (must be absent for native) |
| COMP_PE_OUTPUT_TAG_MISSING | A tagged non-load/store `fabric.pe` is missing the required `output_tag` attribute |
| COMP_LOADPE_TRANSPARENT_NATIVE | A load PE uses transparent mode but has native (untagged) interface |
| COMP_LOADPE_TRANSPARENT_QUEUE_DEPTH | A transparent load PE has nonzero `queue_depth` |
| COMP_STOREPE_TRANSPARENT_NATIVE | A store PE uses transparent mode but has native (untagged) interface |
| COMP_STOREPE_TRANSPARENT_QUEUE_DEPTH | A transparent store PE has nonzero `queue_depth` |
| COMP_ADG_COMBINATIONAL_LOOP | A cycle exists in the connection graph where every element is combinational (zero-delay), causing signal instability |
| COMP_FIFO_DEPTH_ZERO | `fabric.fifo` depth must be >= 1 |
| COMP_FIFO_TYPE_MISMATCH | `fabric.fifo` input and output types do not match |
| COMP_FIFO_INVALID_TYPE | `fabric.fifo` type must be a native value type or `!dataflow.tagged` |
| COMP_FIFO_BYPASSED_NOT_BYPASSABLE | `fabric.fifo` has `bypassed` attribute present without `bypassable` |
| COMP_FIFO_BYPASSED_MISSING | `fabric.fifo` has `bypassable` set but `bypassed` attribute is missing |
| COMP_FANOUT_MODULE_INNER | SSA result of an operation inside `fabric.module` body has multiple consumers; use switch broadcast for data duplication |
| COMP_FANOUT_MODULE_BOUNDARY | Module input argument feeds multiple instance/operation input ports; use switch broadcast for data duplication |
| COMP_OUTPUT_UNCONNECTED | An instance output port has no consumer in the module body |
| COMP_OUTPUT_DANGLING | An instance output port is not connected to a module output |
| COMP_INPUT_UNCONNECTED | An instance input port has no driver |
| COMP_MULTI_DRIVER | An instance input port has multiple drivers |
| COMP_HANDSHAKE_CTRL_MULTI_MEM | A handshake control token feeds multiple memory operations during SCF-to-Handshake conversion |

### Instance Error Examples

```mlir
// ERROR: COMP_PE_INSTANCE_ILLEGAL_TARGET
// Inside fabric.pe, only named fabric.pe targets are legal.
fabric.module @inner(%a: i32, %b: i32) -> (i32) { ... }
fabric.module @top(%a: i32, %b: i32, %c: i32) -> (i32) {
  %r = fabric.pe %a, %b, %c : (i32, i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %out = fabric.instance @inner(%x, %y) : (i32, i32) -> (i32)  // illegal
    %s = arith.addi %out, %z : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

// ERROR: COMP_INSTANCE_OPERAND_MISMATCH
// @alu expects (i32, i32) but only one operand provided
fabric.pe @alu(%a: i32, %b: i32) -> (i32) { ... }
%r = fabric.instance @alu(%x) : (i32) -> (i32)

// ERROR: COMP_INSTANCE_UNRESOLVED
// @nonexistent is not defined
%r = fabric.instance @nonexistent(%x) : (i32) -> (i32)
```

### PE Body Error Examples

```mlir
// ERROR: COMP_PE_EMPTY_BODY
fabric.pe @empty(%a: i32) -> (i32) {
  fabric.yield %a : i32  // no non-terminator operations
}

// ERROR: COMP_PE_MIXED_INTERFACE
// Mixing native (i32) and tagged ports
fabric.pe @mixed(%a: i32, %b: !dataflow.tagged<i32, i4>) -> (i32) { ... }

// ERROR: COMP_PE_DATAFLOW_BODY
// Mixing dataflow with arith
fabric.pe @bad_mix(%a: i32, %b: i32) -> (i32) {
  %c = dataflow.invariant %d, %a : i1, i32 -> i32
  %s = arith.addi %c, %b : i32
  fabric.yield %s : i32
}

// ERROR: COMP_PE_MIXED_CONSUMPTION
// Mixing full-consume (arith.addi) with partial-consume (handshake.mux)
fabric.pe @bad_consume(%sel: i1, %a: i32, %b: i32) -> (i32) {
  %sum = arith.addi %a, %b : i32
  %r = handshake.mux %sel [%sum, %b] : i1, i32, i32 -> i32
  fabric.yield %r : i32
}

// ERROR: COMP_PE_OUTPUT_TAG_NATIVE
// Native PE must not have output_tag
fabric.pe @bad_tag(%a: i32) -> (i32)
    [output_tag = [0 : i4]] { ... }

// ERROR: COMP_PE_OUTPUT_TAG_MISSING
// Tagged non-load/store PE must have output_tag
fabric.pe @no_tag(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
    [latency = [1 : i16, 1 : i16, 1 : i16]] { ... }
```

## CFG_ (Runtime Configuration Errors, Hardware Code)

| Code | Symbol | Condition |
|------|--------|-----------|
| 0 | OK | No error |
| 1 | CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT | Multiple inputs route to the same output (fan-in) |
| 4 | CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT | In one temporal-sw slot, multiple inputs route to the same output (per-slot fan-in) |
| 5 | CFG_TEMPORAL_SW_DUP_TAG | Duplicate tags in `route_table` slots |
| 6 | CFG_TEMPORAL_PE_DUP_TAG | Duplicate tags in `instruction_mem` |
| 7 | CFG_TEMPORAL_PE_ILLEGAL_REG | Register index encodes a value >= `num_register` |
| 8 | CFG_TEMPORAL_PE_REG_TAG_NONZERO | `res_tag != 0` when writing a register |
| 9 | CFG_MAP_TAG_DUP_TAG | `map_tag` has multiple valid entries with the same `src_tag` |
| 10 | CFG_PE_STREAM_CONT_COND_ONEHOT | `dataflow.stream` `cont_cond_sel` register is not one-hot (`<`, `<=`, `>`, `>=`, `!=`) |

## RT_ (Runtime Execution Errors, Hardware Code)

| Code | Symbol | Condition |
|------|--------|-----------|
| 256 | RT_TEMPORAL_PE_NO_MATCH | Input tag matches no instruction |
| 257 | RT_TEMPORAL_SW_NO_MATCH | Input tag matches no route table slot |
| 258 | RT_MAP_TAG_NO_MATCH | `map_tag` finds no valid entry |
| 259 | RT_DATAFLOW_STREAM_ZERO_STEP | `dataflow.stream` observes `step = 0` at runtime |
| 260 | RT_MEMORY_TAG_OOB | A tagged load/store request uses `tag >= count` |
| 261 | RT_MEMORY_STORE_DEADLOCK | A store request cannot be paired with a matching address or data for the same tag within the default timeout (65535 cycles) |
| 262 | RT_SWITCH_UNROUTED_INPUT | A `fabric.switch` input with physical connectivity receives a valid token but has no enabled route |
| 263 | RT_TEMPORAL_SW_UNROUTED_INPUT | A `fabric.temporal_sw` input receives a valid token but the matched route_table slot does not route that input |

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)

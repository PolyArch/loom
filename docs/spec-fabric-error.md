# Fabric Error Codes Specification

## Overview

This document defines the global error code space for Fabric hardware. It is
the single source of truth for error code values. Other documents reference
these symbols by name.

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

CFG_ error codes use the range 0-7. Codes 8-255 are reserved for future use.

RT_ error codes start at 256 and increase sequentially.

## COMP_ (Compile-Time Errors, No Hardware Code)

| Symbol | Condition |
|--------|-----------|
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
| COMP_TEMPORAL_PE_SRC_MISMATCH | A human-readable source uses `in(j)` where `j` is not the operand position |
| COMP_TEMPORAL_PE_TAG_WIDTH | `K != num_bits(T)` or `M != N` for interface `!dataflow.tagged<T, iN>` |
| COMP_TEMPORAL_PE_TAGGED_PE | A `fabric.temporal_pe` contains a tagged `fabric.pe` |
| COMP_TEMPORAL_PE_LOADSTORE | A `fabric.temporal_pe` contains a load/store PE |
| COMP_MAP_TAG_TABLE_SIZE | `table_size` is out of range [1, 256] |
| COMP_MAP_TAG_TABLE_LENGTH | `table` length does not equal `table_size` |
| COMP_ADD_TAG_TYPE_MISMATCH | `fabric.add_tag` result value type does not match input type |
| COMP_DEL_TAG_TYPE_MISMATCH | `fabric.del_tag` output type does not match input tagged value type |
| COMP_MEMORY_PORTS_EMPTY | `ldCount == 0` and `stCount == 0` |
| COMP_MEMORY_LSQ_WITHOUT_STORE | `lsqDepth != 0` when `stCount == 0` |
| COMP_MEMORY_LSQ_MIN | `lsqDepth < 1` when `stCount > 0` |
| COMP_MEMORY_ADDR_TYPE | Address port is not `index` or `!dataflow.tagged<index, iK>` |
| COMP_MEMORY_DATA_TYPE | Data port element type does not match memref element type |
| COMP_MEMORY_TAG_REQUIRED | `ldCount > 1` or `stCount > 1` but ports are not tagged |
| COMP_MEMORY_TAG_FOR_SINGLE | Tagged ports used when `ldCount == 1` or `stCount == 1` |
| COMP_MEMORY_TAG_WIDTH | Tag width is smaller than `log2Ceil(count)` or mismatched across ports |
| COMP_MEMORY_STATIC_REQUIRED | `fabric.memory` uses a dynamic memref type |
| COMP_MEMORY_PRIVATE_OUTPUT | `fabric.module` yields a memref not produced by `fabric.memory` with `private = false` |
| COMP_MEMORY_EXTMEM_BINDING | `fabric.extmemory` memref operand is not a `fabric.module` memref input |
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
| COMP_INSTANCE_ILLEGAL_TARGET | `fabric.instance` references an invalid or unsupported target type (e.g., `fabric.add_tag`, `fabric.map_tag`, `fabric.del_tag`) |
| COMP_INSTANCE_OPERAND_MISMATCH | `fabric.instance` operand count or types do not match the referenced module signature |
| COMP_INSTANCE_RESULT_MISMATCH | `fabric.instance` result count or types do not match the referenced module signature |
| COMP_INSTANCE_UNRESOLVED | `fabric.instance` references a symbol that does not exist |
| COMP_PE_EMPTY_BODY | A `fabric.pe` body contains no non-terminator operations |
| COMP_PE_MIXED_INTERFACE | A `fabric.pe` has mixed native and tagged ports |
| COMP_PE_DATAFLOW_BODY | A `fabric.pe` body contains `dataflow` operations mixed with non-dataflow operations |
| COMP_PE_MIXED_CONSUMPTION | A `fabric.pe` body mixes full-consume and partial-consume operations |
| COMP_PE_OUTPUT_TAG_NATIVE | A native `fabric.pe` has `output_tag` attribute (must be absent for native) |
| COMP_PE_OUTPUT_TAG_MISSING | A tagged non-load/store `fabric.pe` is missing the required `output_tag` attribute |

### Instance Error Examples

```mlir
// ERROR: COMP_INSTANCE_ILLEGAL_TARGET
// fabric.add_tag cannot be instantiated; must be used inline
%t = fabric.instance @my_add_tag(%v) : (i32) -> (!dataflow.tagged<i32, i4>)

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
| 1 | CFG_SWITCH_ROUTE_MULTI_OUT | A single output routes multiple inputs |
| 2 | CFG_SWITCH_ROUTE_MULTI_IN | A single input routes multiple outputs |
| 3 | CFG_TEMPORAL_SW_DUP_TAG | Duplicate tags in `route_table` slots |
| 4 | CFG_TEMPORAL_PE_DUP_TAG | Duplicate tags in `instruction_mem` |
| 5 | CFG_TEMPORAL_PE_ILLEGAL_REG | Register index encodes a value >= `num_register` |
| 6 | CFG_TEMPORAL_PE_REG_TAG_NONZERO | `res_tag != 0` when writing a register |
| 7 | CFG_MAP_TAG_DUP_TAG | `map_tag` has multiple valid entries with the same `src_tag` |

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

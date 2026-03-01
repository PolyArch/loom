# Fabric Error Codes Specification

## Overview

This document defines the global error code space for Fabric hardware.

Authoritative source files for error code definitions:
- **CPL_ codes (C++)**: `include/loom/Hardware/Common/FabricError.h`
- **CFG_/RT_ codes (SV)**: `lib/loom/Hardware/SystemVerilog/Common/fabric_error.svh`

Other documents reference these symbols by name.

Fabric errors are classified into three classes:

- **CPL_**: compile-time errors raised by the Loom compiler. These have no
  hardware error code.
- **CFG_**: runtime configuration errors caused by invalid runtime parameters
  (configuration registers or tables). These are reported by hardware.
- **RT_**: runtime execution errors not caused by configuration. These are
  reported by hardware.

CFG_ errors are detected after writing runtime configuration registers or
tables, before or during execution, and do not require dataflow execution to
surface.

CFG_ error codes currently use the range 0-15. Codes 16-255 are reserved for
future use.

RT_ error codes start at 256 and increase sequentially.

## CPL_ (Compile-Time Errors, No Hardware Code)

| Symbol | Condition |
|--------|-----------|
| CPL_SWITCH_PORT_ZERO | `fabric.switch` has zero inputs or outputs |
| CPL_SWITCH_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| CPL_SWITCH_ROW_EMPTY | A connectivity row has no `1` entries |
| CPL_SWITCH_COL_EMPTY | A connectivity column has no `1` entries |
| CPL_SWITCH_PORT_LIMIT | `fabric.switch` has more than 32 inputs or outputs |
| CPL_SWITCH_ROUTE_LEN_MISMATCH | `route_table` length does not match connected positions |
| CPL_ROUTING_PAYLOAD_NOT_BITS | A routing node port type is not `!dataflow.bits<N>` or `none` |
| CPL_TEMPORAL_SW_PORT_ZERO | `fabric.temporal_sw` has zero inputs or outputs |
| CPL_TEMPORAL_SW_TABLE_SHAPE | `connectivity_table` length is not `num_outputs * num_inputs` |
| CPL_TEMPORAL_SW_ROW_EMPTY | A connectivity row has no `1` entries |
| CPL_TEMPORAL_SW_COL_EMPTY | A connectivity column has no `1` entries |
| CPL_TEMPORAL_SW_PORT_LIMIT | `fabric.temporal_sw` has more than 32 inputs or outputs |
| CPL_TEMPORAL_SW_INTERFACE_NOT_TAGGED | `fabric.temporal_sw` interface type is not tagged |
| CPL_TEMPORAL_SW_NUM_ROUTE_TABLE | `num_route_table` is less than 1 |
| CPL_TEMPORAL_SW_TOO_MANY_SLOTS | `route_table` entries exceed `num_route_table` |
| CPL_TEMPORAL_SW_ROUTE_ILLEGAL | A route entry targets a disconnected position |
| CPL_TEMPORAL_SW_MIXED_FORMAT | `route_table` mixes human-readable and hex entries |
| CPL_TEMPORAL_SW_SLOT_ORDER | Human-readable `route_table` slot indices are not strictly ascending |
| CPL_TEMPORAL_SW_IMPLICIT_HOLE | `route_table` has implicit holes when explicit `invalid` entries exist |
| CPL_TEMPORAL_PE_INTERFACE_NOT_TAGGED | `fabric.temporal_pe` interface type is not tagged |
| CPL_TEMPORAL_PE_NUM_INSTRUCTION | `num_instruction` is less than 1 |
| CPL_TEMPORAL_PE_REG_FIFO_DEPTH | `reg_fifo_depth` is 0 when `num_register > 0`, or nonzero when `num_register = 0` |
| CPL_TEMPORAL_PE_EMPTY_BODY | A `fabric.temporal_pe` contains no FU definitions in its body |
| CPL_TEMPORAL_PE_FU_INVALID | An FU definition references an invalid or unresolvable PE name |
| CPL_TEMPORAL_PE_TAGGED_FU | An FU definition inside `fabric.temporal_pe` has tagged input or output ports |
| CPL_TEMPORAL_PE_FU_ARITY | An FU port count does not match the `fabric.temporal_pe` interface arity |
| CPL_TEMPORAL_PE_REG_DISABLED | An instruction uses `reg(idx)` when `num_register = 0` |
| CPL_TEMPORAL_PE_SRC_MISMATCH | A human-readable source uses `in(j)` where `j` is not the operand position |
| CPL_TEMPORAL_PE_TOO_MANY_SLOTS | `instruction_mem` entries exceed `num_instruction` |
| CPL_TEMPORAL_PE_MIXED_FORMAT | `instruction_mem` mixes human-readable and hex entries |
| CPL_TEMPORAL_PE_SLOT_ORDER | Human-readable `instruction_mem` slot indices are not strictly ascending |
| CPL_TEMPORAL_PE_IMPLICIT_HOLE | `instruction_mem` has implicit holes when explicit `invalid` entries exist |
| CPL_TEMPORAL_PE_DEST_COUNT | Destination count in a human-readable instruction does not equal `num_outputs` |
| CPL_TEMPORAL_PE_SRC_COUNT | Source count in a human-readable instruction does not equal `num_inputs` |
| CPL_TEMPORAL_PE_TAG_WIDTH | `K != num_bits(T)` or tag-width metadata is inconsistent for interface `!dataflow.tagged<T, iJ>` |
| CPL_TEMPORAL_PE_TAGGED_PE | A `fabric.temporal_pe` contains a tagged `fabric.pe` |
| CPL_TEMPORAL_PE_LOADSTORE | A `fabric.temporal_pe` contains a load/store PE |
| CPL_TEMPORAL_PE_DATAFLOW_INVALID | A `fabric.temporal_pe` contains a dataflow PE (carry/invariant/gate/stream) |
| CPL_TEMPORAL_PE_FU_WIDTH | An FU port data width exceeds the `fabric.temporal_pe` interface value type width |
| CPL_MAP_TAG_TABLE_SIZE | `table_size` is out of range [1, 256] |
| CPL_MAP_TAG_TABLE_LENGTH | `table` length does not equal `table_size` |
| CPL_MAP_TAG_VALUE_TYPE_MISMATCH | `fabric.map_tag` output value type does not match input value type |
| CPL_ADD_TAG_VALUE_TYPE_MISMATCH | `fabric.add_tag` result value type does not match input type |
| CPL_ADD_TAG_VALUE_OVERFLOW | `fabric.add_tag` configured `tag` value exceeds the representable range of the output tag type |
| CPL_DEL_TAG_VALUE_TYPE_MISMATCH | `fabric.del_tag` output type does not match input tagged value type |
| CPL_TAG_WIDTH_RANGE | Tag type width is outside the allowed range defined by `!dataflow.tagged` (`i1` to `i16`) |
| CPL_MEMORY_PORTS_EMPTY | `ldCount == 0` and `stCount == 0` |
| CPL_MEMORY_LSQ_WITHOUT_STORE | `lsqDepth != 0` when `stCount == 0` |
| CPL_MEMORY_LSQ_MIN | `lsqDepth < 1` when `stCount > 0` |
| CPL_MEMORY_ADDR_TYPE | Address port is not `bits<ADDR_BIT_WIDTH>` or `!dataflow.tagged<bits<ADDR_BIT_WIDTH>, iK>` |
| CPL_MEMORY_DATA_TYPE | Data port element type does not match memref element type |
| CPL_MEMORY_TAG_REQUIRED | `ldCount > 1` or `stCount > 1` but ports are not tagged |
| CPL_MEMORY_TAG_FOR_SINGLE | Tagged ports used when `ldCount == 1` or `stCount == 1` |
| CPL_MEMORY_TAG_WIDTH | Tag width is smaller than `log2Ceil(count)` or mismatched across ports |
| CPL_MEMORY_STATIC_REQUIRED | `fabric.memory` uses a dynamic memref type |
| CPL_MEMORY_PRIVATE_OUTPUT | `fabric.module` yields a memref not produced by `fabric.memory` with `is_private = false` |
| CPL_MEMORY_EXTMEM_BINDING | `fabric.extmemory` memref operand is not a `fabric.module` memref input |
| CPL_MEMORY_EXTMEM_PRIVATE | `is_private` is supplied on `fabric.extmemory` |
| CPL_MEMORY_INVALID_REGION | `numRegion < 1` on `fabric.memory` or `fabric.extmemory` |
| CPL_FABRIC_TYPE_MISMATCH | A connection inside `fabric.module` uses mismatched types or bit widths without an explicit conversion |
| CPL_MODULE_PORT_ORDER | `fabric.module` ports are not in the required order: memref*, native*, tagged* |
| CPL_MODULE_EMPTY_BODY | `fabric.module` body contains no operations other than the terminator |
| CPL_MODULE_MISSING_YIELD | `fabric.module` body does not end with `fabric.yield` |
| CPL_PE_LOADSTORE_BODY | A load/store PE does not contain exactly one `handshake.load` or `handshake.store` |
| CPL_PE_LOADSTORE_TAG_MODE | Invalid combination of `output_tag`, tagged ctrl, and queue depth attributes |
| CPL_PE_LOADSTORE_TAG_WIDTH | Tag widths do not match across addr/data/ctrl ports |
| CPL_PE_CONSTANT_BODY | A constant PE contains operations other than a single `handshake.constant` |
| CPL_PE_INSTANCE_ONLY_BODY | A `fabric.pe` body contains only a single `fabric.instance` with no other operations (exempt inside `fabric.temporal_pe`) |
| CPL_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE | `operand_buffer_size` is set when `enable_share_operand_buffer = false` |
| CPL_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING | `operand_buffer_size` is missing when `enable_share_operand_buffer = true` |
| CPL_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE | `operand_buffer_size` is out of range [1, 8192] |
| CPL_PE_INSTANCE_ILLEGAL_TARGET | Inside `fabric.pe`, `fabric.instance` targets a non-PE operation (only named `fabric.pe` targets are legal) |
| CPL_INSTANCE_OPERAND_MISMATCH | `fabric.instance` operand count or types do not match the referenced module signature |
| CPL_INSTANCE_RESULT_MISMATCH | `fabric.instance` result count or types do not match the referenced module signature |
| CPL_INSTANCE_UNRESOLVED | `fabric.instance` references a symbol that does not exist |
| CPL_INSTANCE_CYCLIC_REFERENCE | `fabric.instance` forms a cyclic reference in a scope that requires acyclic instantiation (for example, inside `fabric.pe`) |
| CPL_PE_EMPTY_BODY | A `fabric.pe` body contains no non-terminator operations |
| CPL_PE_MIXED_INTERFACE | A `fabric.pe` has mixed native and tagged ports |
| CPL_PE_TAGGED_INTERFACE_NATIVE_PORTS | A `fabric.pe` with tagged interface category has native (untagged) ports |
| CPL_PE_NATIVE_INTERFACE_TAGGED_PORTS | A `fabric.pe` with native interface category has tagged ports |
| CPL_PE_DATAFLOW_BODY | A `fabric.pe` dataflow body violates dataflow exclusivity: mixed dataflow/non-dataflow ops, multiple dataflow ops, or any `fabric.instance` nesting |
| CPL_PE_MIXED_CONSUMPTION | A `fabric.pe` body mixes full-consume and partial-consume operations |
| CPL_PE_OUTPUT_TAG_NATIVE | A native `fabric.pe` has `output_tag` attribute (must be absent for native) |
| CPL_PE_OUTPUT_TAG_MISSING | A tagged non-load/store `fabric.pe` is missing the required `output_tag` attribute |
| CPL_LOADPE_TRANSPARENT_NATIVE | A load PE uses transparent mode but has native (untagged) interface |
| CPL_LOADPE_TRANSPARENT_QUEUE_DEPTH | A transparent load PE has nonzero `queue_depth` |
| CPL_STOREPE_TRANSPARENT_NATIVE | A store PE uses transparent mode but has native (untagged) interface |
| CPL_STOREPE_TRANSPARENT_QUEUE_DEPTH | A transparent store PE has nonzero `queue_depth` |
| CPL_ADG_COMBINATIONAL_LOOP | A cycle exists in the connection graph where every element is combinational (zero-delay), causing signal instability |
| CPL_FIFO_DEPTH_ZERO | `fabric.fifo` depth must be >= 1 |
| CPL_FIFO_TYPE_MISMATCH | `fabric.fifo` input and output types do not match |
| CPL_FIFO_INVALID_TYPE | `fabric.fifo` type must be a native value type or `!dataflow.tagged` |
| CPL_FIFO_BYPASSED_NOT_BYPASSABLE | `fabric.fifo` has `bypassed` attribute present without `bypassable` |
| CPL_FIFO_BYPASSED_MISSING | `fabric.fifo` has `bypassable` set but `bypassed` attribute is missing |
| CPL_FANOUT_MODULE_INNER | SSA result of an operation inside `fabric.module` body has multiple consumers; use switch broadcast for data duplication |
| CPL_FANOUT_MODULE_BOUNDARY | Module input argument feeds multiple instance/operation input ports; use switch broadcast for data duplication |
| CPL_OUTPUT_UNCONNECTED | An instance output port has no consumer in the module body |
| CPL_OUTPUT_DANGLING | An instance output port is not connected to a module output |
| CPL_INPUT_UNCONNECTED | An instance input port has no driver |
| CPL_MULTI_DRIVER | An instance input port has multiple drivers |
| CPL_HANDSHAKE_CTRL_MULTI_MEM | A handshake control token feeds multiple memory operations during SCF-to-Handshake conversion |

### Instance Error Examples

```mlir
// ERROR: CPL_PE_INSTANCE_ILLEGAL_TARGET
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

// ERROR: CPL_INSTANCE_OPERAND_MISMATCH
// @alu expects (i32, i32) but only one operand provided
fabric.pe @alu(%a: i32, %b: i32) -> (i32) { ... }
%r = fabric.instance @alu(%x) : (i32) -> (i32)

// ERROR: CPL_INSTANCE_UNRESOLVED
// @nonexistent is not defined
%r = fabric.instance @nonexistent(%x) : (i32) -> (i32)
```

### PE Body Error Examples

```mlir
// ERROR: CPL_PE_EMPTY_BODY
fabric.pe @empty(%a: i32) -> (i32) {
  fabric.yield %a : i32  // no non-terminator operations
}

// ERROR: CPL_PE_MIXED_INTERFACE
// Mixing native (i32) and tagged ports
fabric.pe @mixed(%a: i32, %b: !dataflow.tagged<i32, i4>) -> (i32) { ... }

// ERROR: CPL_PE_DATAFLOW_BODY
// Mixing dataflow with arith
fabric.pe @bad_mix(%a: i32, %b: i32) -> (i32) {
  %c = dataflow.invariant %d, %a : i1, i32 -> i32
  %s = arith.addi %c, %b : i32
  fabric.yield %s : i32
}

// ERROR: CPL_PE_MIXED_CONSUMPTION
// Mixing full-consume (arith.addi) with partial-consume (handshake.mux)
fabric.pe @bad_consume(%sel: i1, %a: i32, %b: i32) -> (i32) {
  %sum = arith.addi %a, %b : i32
  %r = handshake.mux %sel [%sum, %b] : i1, i32, i32 -> i32
  fabric.yield %r : i32
}

// ERROR: CPL_PE_OUTPUT_TAG_NATIVE
// Native PE must not have output_tag
fabric.pe @bad_tag(%a: i32) -> (i32)
    [output_tag = [0 : i4]] { ... }

// ERROR: CPL_PE_OUTPUT_TAG_MISSING
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
| 11 | CFG_PE_CMPI_PREDICATE_INVALID | PE cmpi predicate field value >= 10 (only 0-9 are valid for integer comparison) |
| 12 | CFG_MEMORY_OVERLAP_TAG_REGION | `fabric.memory` addr_offset_table has overlapping tag ranges between valid regions |
| 13 | CFG_MEMORY_EMPTY_TAG_RANGE | `fabric.memory` addr_offset_table has a region with `end_tag <= start_tag` (empty half-open range) |
| 14 | CFG_EXTMEMORY_OVERLAP_TAG_REGION | `fabric.extmemory` addr_offset_table has overlapping tag ranges between valid regions (half-open interval overlap) |
| 15 | CFG_EXTMEMORY_EMPTY_TAG_RANGE | `fabric.extmemory` addr_offset_table has a region with `end_tag <= start_tag` (empty half-open range) |

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
| 264 | RT_MEMORY_NO_MATCH | A `fabric.memory` load/store tag matches no valid region in the addr_offset_table |
| 265 | RT_EXTMEMORY_NO_MATCH | A `fabric.extmemory` load/store tag matches no valid region in the addr_offset_table |

## Same-Cycle Error Precedence

For any module that can detect multiple error conditions simultaneously
(`fabric.switch`, `fabric.temporal_sw`, `fabric.memory`, `fabric.extmemory`),
the error with the **numerically smallest error code** is captured first. This
establishes a fixed priority: configuration errors (CFG\_, codes 1-255) always
take precedence over runtime execution errors (RT\_, codes 256+). Within the
same error class, the lower code number wins.

Once `error_valid` is asserted, it is sticky (remains asserted until reset).
Later errors do not overwrite the first captured error code.

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-tag.md](./spec-fabric-tag.md)
- [spec-fabric-pe.md](./spec-fabric-pe.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)

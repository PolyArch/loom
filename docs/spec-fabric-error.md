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
| COMP_MAP_TAG_TABLE_SIZE | `table_size` is out of range [1, 256] |
| COMP_MAP_TAG_TABLE_LENGTH | `table` length does not equal `table_size` |

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

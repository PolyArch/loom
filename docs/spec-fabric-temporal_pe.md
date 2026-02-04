# Fabric Temporal PE Specification

## Overview

A `fabric.temporal_pe` is a time-multiplexed processing element that executes
one of multiple functional unit (FU) types based on the tag carried by input
values. The tag is used for instruction matching rather than direct indexing.

Each instruction slot stores a tag value and an opcode that selects a FU type.
When a tagged token arrives, its tag selects the instruction whose tag matches
that value. If no instruction matches, the hardware signals a runtime error.
If multiple instructions could match, the configuration is invalid because it
contains duplicate tags.

## Operation: `fabric.temporal_pe`

### Syntax

```
fabric.temporal_pe @name(
  %in0: !dataflow.tagged<T, iN>, %in1: !dataflow.tagged<T, iN>, ...
) -> (!dataflow.tagged<T, iN>, ...)
  [num_register = R, num_instruction = I, num_instance = F]
  {instruction_mem = [ ... ]} {
  // FU definitions
  fabric.yield %fu0_out0, %fu0_out1, %fu1_out0, %fu1_out1, ...
}
```

### Interface Types

- All inputs and outputs must be `!dataflow.tagged` types.
- All ports must use the same tagged type.
- Tag width must be in the range `i1` to `i16`.

### FU Types and Body Structure

The body of `fabric.temporal_pe` defines FU types. Each FU type is represented
by a `fabric.pe` or by a `fabric.instance` of a named `fabric.pe`.

Constraints:

- Each FU type must have the same number of inputs and outputs as the
  `fabric.temporal_pe` itself.
- Each FU type operates on value-only data. Tags are stripped at the boundary.
- The body may contain only FU definitions (`fabric.pe` or `fabric.instance`)
  and a single `fabric.yield`.
- Any `fabric.instance` inside `fabric.temporal_pe` must reference a
  `fabric.pe`.
- `fabric.switch` is not allowed inside `fabric.temporal_pe`.
- `fabric.temporal_sw` is not allowed inside `fabric.temporal_pe`.

### Yield Ordering

The `fabric.temporal_pe` terminator yields `num_fu_types * num_outputs` values.
The ordering is:

- FU types appear in the order they are defined in the body.
- For each FU type, outputs appear in ascending output index order.

### Attributes

#### `num_register` (hardware parameter)

- Unsigned integer.
- Number of internal registers for inter-instruction communication.
 - Must be greater than or equal to 0.

#### `num_instruction` (hardware parameter)

- Unsigned integer.
- Maximum number of instruction slots.
- Must be greater than 0.

#### `num_instance` (hardware parameter)

- Unsigned integer.
- FIFO depth for each internal register.
- Must be 0 if `num_register` is 0.
- Must be at least 1 if `num_register` is greater than 0.

#### `instruction_mem` (runtime configuration parameter)

- Array of instruction slot entries.
- Length must be less than or equal to `num_instruction`.
- Supports human-readable and machine (hex) formats.
- All entries in the array must use the same format.

The number of FU types defined in the body is independent of
`num_instruction`. Instruction slots select among FU types via the opcode.

### Tag Matching Semantics

- Each valid instruction slot has an associated tag value.
- At runtime, an input tag is matched against instruction tags.
- Exactly one instruction must match a given input tag.
- Duplicate tags in the instruction memory are configuration errors.

### Operand Buffer Semantics

Each instruction slot contains per-input operand buffers described by
`op_valid` and `op_value` fields in the machine format. These fields are
runtime state, not static instruction encoding.

At reset or configuration time, all `op_valid` bits are initialized to `0`
and `op_value` is initialized to `0`.

When a tagged token arrives on input `i`:

1. The tag selects a unique instruction slot `s`. If no slot matches, a
   runtime error is raised. If multiple slots match, the configuration is
   invalid due to duplicate tags.
2. If `op_valid` for operand `i` in slot `s` is `0`, the token is consumed,
   its value bits are stored into `op_value`, and `op_valid` is set to `1`.
3. If `op_valid` for operand `i` in slot `s` is already `1`, the input is
   backpressured and the token is not consumed.

An instruction in slot `s` may fire when all operands are ready:

- For each input operand `i`, either `op_is_reg = 1` (operand reads from a
  register) or `op_valid = 1` (operand buffered from an input).

When the instruction fires:

- Input-buffered operands are consumed and their `op_valid` bits are cleared.
- Register operands dequeue one element from their FIFOs.
- Result values are produced and routed to outputs or registers.

At most one instruction may fire in a single cycle. The temporal PE does not
issue multiple instructions in parallel.

### IMPORTANT: Potential Deadlock

Because operand buffers are per-slot, the temporal PE can deadlock if some
operands for a tag never arrive or if inputs are blocked behind full operand
buffers. This is a known limitation and requires a system-level solution.

### Output Tag Semantics

- Output tags are taken from `instruction_mem` result fields.
- Tags generated by FUs are ignored.
- If a `fabric.pe` inside a temporal PE has `output_tag`, the value is ignored
  and the compiler emits a warning.

## Instruction Memory Format

`instruction_mem` provides per-slot runtime configuration. Each slot stores:

- A match tag.
- An opcode selecting a FU type.
- Operand source selections.
- Result destination selections and output tags.

The slot index is the storage location only. Tag matching determines the
active instruction at runtime.

### Sparse Format Rules

The following rules apply to human-readable entries:

- Slot indices must be strictly ascending.
- Implicit holes are allowed only when there are no explicit `invalid` entries.
- If any explicit `invalid` entry is present, all holes must be explicit.
- Trailing invalid entries may be omitted.

For machine format entries:

- The array is dense by index.
- Only trailing `0x0` entries may be omitted.

### Human-Readable Format

Each entry is a string with the following syntax:

```
inst[slot]: when(tag=TAG) DESTS = NAME(opcode) SRCS
```

- `slot` is the instruction memory index.
- `TAG` is the tag value to match.
- `NAME` is an informational mnemonic. Hardware uses only the numeric opcode.
- `opcode` selects a FU type.
- `DESTS` is a comma-separated list of destinations.
- `SRCS` is a comma-separated list of sources.

#### Destinations

Each destination is one of:

- `out(idx, tag=VALUE)`
- `reg(idx, tag=VALUE)`

`tag=VALUE` is optional. If omitted, the output tag defaults to the match tag.

If the destination is `reg(idx)`, the tag must be omitted or set to `0`. A
nonzero tag for a register destination is a configuration error.

The number of destinations must equal the number of outputs of the temporal PE.

#### Sources

Each source is one of:

- `in(idx)`
- `reg(idx)`

The number of sources must equal the number of inputs of the temporal PE.

#### Invalid Slot

```
inst[slot]: invalid
```

#### Example

```
instruction_mem = [
  "inst[0]: when(tag=3) out(0, tag=1) = add(0) in(0), in(1)",
  "inst[1]: when(tag=4) out(0) = mul(1) in(0), reg(0)",
  "inst[2]: invalid"
]
```

### Machine Format (Hex)

Each entry is a hexadecimal string:

```
0x<hex_value>
```

In machine format, `op_valid` and `op_value` bits represent runtime operand
buffers. Configuration should set `op_valid = 0` and `op_value = 0`.

Bit layout is from LSB to MSB:

```
| valid | tag | opcode | operand fields ... | result fields ... |
```

Definitions:

- `valid`: 1 bit. `0` means invalid slot.
- `tag`: `M` bits, where `M` is the tag width.
- `opcode`: `O` bits, where `O = log2Ceil(num_fu_types)`.

Operand field layout:

```
| op_valid | op_is_reg | op_reg_idx | op_value |
```

- `op_valid`: 1 bit. Indicates whether the operand buffer holds a valid value.
- `op_is_reg` and `op_reg_idx` are present only if `num_register > 0`.
- `op_reg_idx`: `log2(num_register)` bits.
- `op_value`: `K` bits, where `K` is the value bit width.

`op_valid` and `op_value` are runtime operand buffers. When the source is
`in(idx)` and `op_is_reg = 0`, the input value is stored into `op_value` and
`op_valid` is set to `1` until the instruction fires. When the source is
`reg(idx)` and `op_is_reg = 1`, `op_valid` and `op_value` are ignored and
should be set to `0`.

Result field layout:

```
| res_is_reg | res_reg_idx | res_tag |
```

- `res_is_reg` and `res_reg_idx` are present only if `num_register > 0`.
- `res_reg_idx`: `log2(num_register)` bits.
- `res_tag`: `M` bits, the output tag for this result.

There is no explicit result-valid bit. Every instruction must provide one
result field per output.

### Operand and Result Ordering

Operands and results are laid out by increasing index from LSB to MSB:

- `operand[0]` is closest to the `opcode` field (lowest bits after `opcode`).
- `operand[1]` follows `operand[0]`, and so on.
- `result[0]` follows the operand block.
- `result[N-1]` is closest to the MSB.

Example layout for `L = 2`, `N = 1`, `R = 0`, `M = 3`, `K = 8`, `O = 2`:

```
| valid | tag[2:0] | opcode[1:0] | op0[8:0] | op1[8:0] | res0[2:0] |
```

### Width Formulas

Let:

- `L` = number of inputs
- `N` = number of outputs
- `R` = `num_register`
- `M` = tag bit width
- `K` = value bit width
- `O` = `log2Ceil(num_fu_types)`

Then:

- `operand_width = 1 + (R > 0 ? 1 + log2(R) : 0) + K`
- `result_width = (R > 0 ? 1 + log2(R) : 0) + M`
- `instruction_width = 1 + M + O + L * operand_width + N * result_width`

### Opcode Assignment

- FU types are indexed in the order they appear in the body.
- The opcode value selects the FU type.
- If there is only one FU type, `O = 0` and the opcode field is omitted.

## Register Semantics

- Each register is a FIFO with depth `num_instance`.
- Writing to a register enqueues one value.
- Reading from a register dequeues one value.
- A register may have multiple readers.
- A register must have only one writer.
- Registers store values only. Tags are not stored.
- If `res_is_reg = 1`, the corresponding `res_tag` must be `0`.
  A nonzero `res_tag` when writing a register is a configuration error.

When a register has multiple readers, it behaves like an internal fork. The
FIFO entry is retained until each dependent instruction has fired once using
that entry. Only after all readers consume the entry is it dequeued.

## Error Conditions

The temporal PE raises a hardware error if any of the following occurs:

- An input tag matches no instruction.
- Duplicate instruction tags are configured (configuration error).
- An instruction specifies illegal register indices.
- An operand or result selects a register while `num_register = 0`.
- A result writes to a register with `res_tag != 0`.

Errors are reported through a hardware-valid error signal and an error code
propagated to the top level.

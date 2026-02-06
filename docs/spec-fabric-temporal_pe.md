# Fabric Temporal PE Specification

## Overview

A `fabric.temporal_pe` is a time-multiplexed processing element that executes
one of multiple functional unit (FU) types based on the tag carried by input
values. The tag is used for instruction matching rather than direct indexing.

Each instruction slot stores a tag value and an opcode that selects a FU type.
When a tagged token arrives, its tag selects the instruction whose tag matches
that value. If no instruction matches, the hardware signals a runtime error
(`RT_TEMPORAL_PE_NO_MATCH`). If multiple instructions could match, the
configuration is invalid because it contains duplicate tags
(`CFG_TEMPORAL_PE_DUP_TAG`). See [spec-fabric-error.md](./spec-fabric-error.md).

## Operation: `fabric.temporal_pe`

### Syntax

```
fabric.temporal_pe @name(
  %in0: !dataflow.tagged<T, iJ>, %in1: !dataflow.tagged<T, iJ>, ...
) -> (!dataflow.tagged<T, iJ>, ...)
  [num_register = R, num_instruction = I, num_instance = F,
   enable_share_operand_buffer = false, operand_buffer_size = S]
  {instruction_mem = [ ... ]} {
  // FU definitions
  fabric.yield %fu0_out0, %fu0_out1, %fu1_out0, %fu1_out1, ...
}
```

### Interface Types

- All inputs and outputs must be `!dataflow.tagged` types.
- All ports must use the same tagged type.
- Tag width must be in the range `i1` to `i16`.

Violations of tagged interface width requirements are compile-time errors:
`COMP_TEMPORAL_PE_TAG_WIDTH`. See [spec-fabric-error.md](./spec-fabric-error.md).

### FU Types and Body Structure

The body of `fabric.temporal_pe` defines FU types. Each FU type is represented
by a `fabric.pe` or by a `fabric.instance` of a named `fabric.pe`.

Constraints:

- Each FU type must have the same number of inputs and outputs as the
  `fabric.temporal_pe` itself.
- Each FU type operates on value-only data. Tags are stripped at the boundary.
- Each FU type must use a native (non-tagged) `fabric.pe` interface.
- Each FU type's value types must match the `fabric.temporal_pe` interface
  value type `T`. For interface `!dataflow.tagged<T, iJ>`, all FU ports must
  use type `T`.
- The body may contain only FU definitions (`fabric.pe` or `fabric.instance`)
  and a single `fabric.yield`.
- Any `fabric.instance` inside `fabric.temporal_pe` must reference a
  `fabric.pe`.
- `fabric.switch` is not allowed inside `fabric.temporal_pe`.
- `fabric.temporal_sw` is not allowed inside `fabric.temporal_pe`.
- Load/store PEs (a `fabric.pe` containing `handshake.load` or `handshake.store`)
  are not allowed inside `fabric.temporal_pe`
  (`COMP_TEMPORAL_PE_LOADSTORE`).

Using a tagged `fabric.pe` inside `fabric.temporal_pe` is a compile-time error
(`COMP_TEMPORAL_PE_TAGGED_PE`). See [spec-fabric-error.md](./spec-fabric-error.md).

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
- Violations are compile-time errors: `COMP_TEMPORAL_PE_NUM_INSTRUCTION`. See
  [spec-fabric-error.md](./spec-fabric-error.md).

#### `num_instance` (hardware parameter)

- Unsigned integer.
- FIFO depth for each internal register.
- Must be 0 if `num_register` is 0.
- Must be at least 1 if `num_register` is greater than 0.
- Violations are compile-time errors: `COMP_TEMPORAL_PE_NUM_INSTANCE`. See
  [spec-fabric-error.md](./spec-fabric-error.md).

#### `enable_share_operand_buffer` (hardware parameter)

- Boolean (default: `false`).
- Selects the operand buffer hardware mode:
  - `false`: Per-instruction operand buffer mode (Mode A)
  - `true`: Shared operand buffer mode (Mode B)
- This is a static hardware parameter determined at synthesis time.
- See [Operand Buffer Architecture](#operand-buffer-architecture) for details.

#### `operand_buffer_size` (hardware parameter)

- Unsigned integer.
- Only valid when `enable_share_operand_buffer = true`.
- Specifies the depth of the shared operand buffer.
- Must be absent or unset when `enable_share_operand_buffer = false`.
- Must be in range [1, 8192] when `enable_share_operand_buffer = true`.
- Violations are compile-time errors:
  - `COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE`: Mode A cannot have size
  - `COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING`: Mode B requires size
  - `COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE`: Size out of [1, 8192] range
- See [spec-fabric-error.md](./spec-fabric-error.md).

#### `instruction_mem` (runtime configuration parameter)

- Array of instruction slot entries.
- Length must be less than or equal to `num_instruction`.
- Supports human-readable and machine (hex) formats.
- All entries in the array must use the same format.

The number of FU types defined in the body is independent of
`num_instruction`. Instruction slots select among FU types via the opcode.
For the formal `config_mem` definition (32-bit word width, depth calculation,
and per-module packing/alignment), see
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

### Tag Matching Semantics

- Each valid instruction slot has an associated tag value.
- At runtime, an input tag is matched against instruction tags.
- Exactly one instruction must match a given input tag.
- Duplicate tags in the instruction memory are configuration errors
  (`CFG_TEMPORAL_PE_DUP_TAG`). See [spec-fabric-error.md](./spec-fabric-error.md).

## Operand Buffer Architecture

The operand buffer is a **separate hardware component** from `instruction_mem`.
It stores incoming operand values until all operands for an instruction are
ready. The operand buffer is internal runtime state, **not part of config_mem**.

Two hardware modes are available, selected by `enable_share_operand_buffer`:

### Mode A: Per-Instruction Operand Buffer (Default)

When `enable_share_operand_buffer = false` (default):

**Structure:**
- Size: `num_instruction` rows x `num_input` columns x `(1 + K)` bits per entry
- Each instruction slot has a dedicated operand buffer row
- Each entry contains: `op_valid` (1 bit) + `op_value` (K bits)

**Operand Arrival:**
1. Incoming operand's tag matches an instruction in `instruction_mem`
2. If no match: runtime error `RT_TEMPORAL_PE_NO_MATCH`
3. If match found at slot `s`:
   - If `op_valid[s][i] = 0`: store value, set `op_valid = 1`
   - If `op_valid[s][i] = 1`: backpressure (block input)

**Instruction Firing:**
- Instruction at slot `s` fires when all operands ready:
  - For each input `i`: either `op_is_reg = 1` OR `op_valid[s][i] = 1`
- After firing: all `op_valid` in row `s` become `0`, `op_value` retained
- At most one instruction fires per cycle

**Deadlock Risk:**
Mode A can deadlock if operands for different tags interleave and block each
other. Use Mode B or upstream scheduling constraints to mitigate.

### Mode B: Shared Operand Buffer

When `enable_share_operand_buffer = true`:

This mode implements per-tag virtual channels using a shared buffer, similar
to virtual channel techniques in network-on-chip designs.

**Structure:**
- Depth: `operand_buffer_size` entries (max 8192)
- Each entry format:

```
| position | tag | operand[0] | operand[1] | ... | operand[L-1] |
```

- `position`: `log2Ceil(operand_buffer_size)` bits (max 13 bits)
- `tag`: J bits (tag width)
- Each operand: `op_valid` (1 bit) + `op_value` (K bits)
- Total entry width: `log2Ceil(S) + J + L * (1 + K)` bits

**Entry Validity:**
- **Valid entry**: at least one `op_valid = 1`
- **Invalid entry**: all `op_valid = 0` (does not count toward capacity)

**Operand Arrival:**
1. Find all entries with matching tag
2. If no matching entries exist:
   - Create new entry with `position = 0`, store operand
3. If matching entries exist:
   - Find entry with largest `position`
   - If that entry's `op_valid[i] = 0`: store operand in that entry
   - If that entry's `op_valid[i] = 1`: create new entry with
     `position = max_position + 1`, store operand
4. If buffer full (see below): backpressure input

**Buffer Full Condition (per input column `i`):**
- All entries have `op_valid[i] = 1`, OR
- Entries with `op_valid[i] = 0` exist, but belong to valid entries (have
  some `op_valid = 1`) with non-matching tags

**Instruction Firing:**
1. Match instruction tag against buffer entries
2. Find entry with matching tag AND `position = 0` AND all `op_valid = 1`
3. Fire instruction using operand values
4. After firing:
   - Fired entry: all `op_valid` become `0` (entry becomes invalid)
   - Other tag-matched entries: `position -= 1`

**Deadlock Mitigation:**
Mode B provides per-tag FIFO ordering, preventing head-of-line blocking between
different tags. However, `operand_buffer_size` must be sufficient for the
expected operand interleaving depth. Formal analysis of required buffer depth
is application-dependent.

### Operand Buffer Initialization

At reset, all operand buffer entries are initialized:
- All `op_valid` bits = 0
- All `op_value` bits = 0 (don't care, controlled by `op_valid`)
- Mode B: all `position` and `tag` fields = 0

### Output Tag Semantics

- Output tags are taken from `instruction_mem` result fields.
- FU types are native and do not generate tags.

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
nonzero tag for a register destination is a configuration error. If
`num_register = 0`, `reg(idx)` destinations are invalid. See
`CFG_TEMPORAL_PE_REG_TAG_NONZERO` and `COMP_TEMPORAL_PE_REG_DISABLED` in
[spec-fabric-error.md](./spec-fabric-error.md).

The number of destinations must equal the number of outputs of the temporal PE.

#### Sources

Each source is one of:

- `in(idx)`
- `reg(idx)`

The number of sources must equal the number of inputs of the temporal PE.
If `num_register = 0`, `reg(idx)` sources are invalid.

Sources are positional. For operand position `i`, the source must be either
`in(i)` or `reg(idx)`. Using `in(j)` where `j != i` is invalid.

Violations are compile-time errors: `COMP_TEMPORAL_PE_REG_DISABLED` and
`COMP_TEMPORAL_PE_SRC_MISMATCH`. See [spec-fabric-error.md](./spec-fabric-error.md).

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

Bit layout is from LSB to MSB:

```
| valid | tag | opcode | operand fields ... | result fields ... |
```

ASCII diagram:

```
+----------------------------------------------------------------------------------+
|                     INSTRUCTION WORD (LSB -> MSB)                                |
+--------+---------+--------+-------------------------+----------------------------+
| valid  | tag[J]  | opcode | operands (L blocks)     | results (N blocks)         |
+--------+---------+--------+-------------------------+----------------------------+
```

Definitions:

- `valid`: 1 bit. `0` means invalid slot.
- `tag`: `J` bits, where `J` is the tag width.
- `opcode`: `O` bits, where `O = log2Ceil(num_fu_types)`.

Operand field layout:

```
| op_is_reg | op_reg_idx |
```

ASCII diagram:

```
+-----------+---------------------------+
| op_is_reg | op_reg_idx (if R > 0)     |
+-----------+---------------------------+
```

- `op_is_reg`: 1 bit. `1` means operand comes from register, `0` means from input.
- `op_reg_idx`: `log2Ceil(num_register)` bits. Present only if `num_register > 0`.
- When `op_is_reg = 0`, the operand comes from the corresponding input port
  and is buffered in the operand buffer (see [Operand Buffer Architecture]).
- When `op_is_reg = 1`, the operand comes from register `op_reg_idx`.

**Note:** `op_valid` and `op_value` are NOT part of `instruction_mem`. They
reside in the separate operand buffer hardware. See
[Operand Buffer Architecture](#operand-buffer-architecture).

Result field layout:

```
| res_is_reg | res_reg_idx | res_tag |
```

ASCII diagram:

```
+-----------+------------+----------------+
| res_is_reg| res_reg_idx| res_tag (J bits)|
+-----------+------------+----------------+
```

- `res_is_reg` and `res_reg_idx` are present only if `num_register > 0`.
- `res_reg_idx`: `log2Ceil(num_register)` bits.
- `res_tag`: `J` bits, the output tag for this result.

There is no explicit result-valid bit. Every instruction must provide one
result field per output.

### Format Selection Guidance

- Use the human-readable format for documentation and inspection.
- Use the machine (hex) format for hardware configuration files and simulation.

### Operand and Result Ordering

Operands and results are laid out by increasing index from LSB to MSB:

- `operand[0]` is closest to the `opcode` field (lowest bits after `opcode`).
- `operand[1]` follows `operand[0]`, and so on.
- `result[0]` follows the operand block.
- `result[N-1]` is closest to the MSB.

#### Base Example 1: Field Layout Illustration (`R = 0`, `O = 2`)

Example layout for `L = 2`, `N = 1`, `R = 0`, `J = 3`, `O = 2`:

```
| valid | tag[2:0] | opcode[1:0] | res0_tag[2:0] |
```

Note: When `R = 0`, operand fields are omitted entirely (0 bits each).
The `op_is_reg` bit is not needed because there are no registers to select.

#### Base Example 2: Concrete Encoding (`R = 0`, `num_fu_types = 2`)

Complete example bitmap (LSB -> MSB):

Parameters:
- `L = 2`, `N = 1`, `R = 0`, `T = i8` (`K = 8`), `tag = i4` (`J = 4`)
- `num_fu_types = 2` (`O = 1`)

Example instruction:
- `valid = 1`
- `tag = 3` (`0011`)
- `opcode = 1`
- `res0_tag = 3` (`0011`)

Bitmap (fields separated by `|`):

```
1 | 0011 | 1 | 0011
```

Hex encoding (10 bits, LSB -> MSB):

```
0x0E7
```

Note: Operand values (`op_valid`, `op_value`) are stored in the separate
operand buffer, not in `instruction_mem`.

#### Complex Example 1: Registers and Multiple Results (`R > 0`, `O > 0`)

Parameters:
- `L = 2`, `N = 2`, `R = 4`, `J = 3`, `num_fu_types = 4` (`O = 2`)
- `operand_config_width = 1 + log2Ceil(4) = 3`
- `result_width = (1 + log2Ceil(4)) + 3 = 6`
- `instruction_width = 1 + 3 + 2 + 2*3 + 2*6 = 24`

Example instruction:
- `valid = 1`
- `tag = 5` (`101`)
- `opcode = 2` (`10`)
- `operand[0] = reg(2)` -> `op0_is_reg = 1`, `op0_reg_idx = 10`
- `operand[1] = in(1)` -> `op1_is_reg = 0`, `op1_reg_idx = 00` (ignored)
- `result[0] = out(0, tag=6)` -> `res0_is_reg = 0`, `res0_reg_idx = 00`, `res0_tag = 110`
- `result[1] = reg(3)` -> `res1_is_reg = 1`, `res1_reg_idx = 11`, `res1_tag = 000`

Bitmap (LSB -> MSB):

```
1 | 101 | 10 | 1 10 | 0 00 | 0 00 110 | 1 11 000
```

Hex encoding (24-bit slot):

```
0x1F016B
```

#### Complex Example 2: Single FU Type with Three Operands (`O = 0`)

Parameters:
- `L = 3`, `N = 1`, `R = 2`, `J = 4`, `num_fu_types = 1` (`O = 0`, opcode omitted)
- `operand_config_width = 1 + log2Ceil(2) = 2`
- `result_width = (1 + log2Ceil(2)) + 4 = 6`
- `instruction_width = 1 + 4 + 0 + 3*2 + 1*6 = 17`

Example instruction:
- `valid = 1`
- `tag = 9` (`1001`)
- `operand[0] = in(0)` -> `op0_is_reg = 0`, `op0_reg_idx = 0` (ignored)
- `operand[1] = reg(1)` -> `op1_is_reg = 1`, `op1_reg_idx = 1`
- `operand[2] = reg(0)` -> `op2_is_reg = 1`, `op2_reg_idx = 0`
- `result[0] = out(0, tag=12)` -> `res0_is_reg = 0`, `res0_reg_idx = 0`, `res0_tag = 1100`

Bitmap (LSB -> MSB):

```
1 | 1001 | 0 0 | 1 1 | 1 0 | 0 0 1100
```

Hex encoding (17-bit slot):

```
0x18393
```

### Width Formulas

Let:

- `L` = number of inputs
- `N` = number of outputs
- `R` = `num_register`
- `J` = tag bit width
- `K` = value bit width
- `O` = `log2Ceil(num_fu_types)`
- `S` = `operand_buffer_size` (Mode B only)

**Instruction Memory Width (config_mem):**

- `operand_config_width = (R > 0 ? 1 + log2Ceil(R) : 0)`
  - Only contains `op_is_reg` and `op_reg_idx`
  - Does NOT include `op_valid` or `op_value` (those are in operand buffer)
- `result_width = (R > 0 ? 1 + log2Ceil(R) : 0) + J`
- `instruction_width = 1 + J + O + L * operand_config_width + N * result_width`

**Operand Buffer Width (internal, not config_mem):**

Mode A (per-instruction):
- `operand_buffer_entry_width = 1 + K` (op_valid + op_value)
- `total_operand_buffer_bits = num_instruction * L * (1 + K)`

Mode B (shared):
- `position_width = log2Ceil(S)` (max 13 bits when S = 8192)
- `operand_buffer_entry_width = position_width + J + L * (1 + K)`
- `total_operand_buffer_bits = S * operand_buffer_entry_width`

For a temporal PE with interface `!dataflow.tagged<T, iJ>`, the bit widths are
defined as:

- `K = num_bits(T)`
- `J = num_bits(iJ)`

Any mismatch between the interface type and these widths is a compile-time
error. See `COMP_TEMPORAL_PE_TAG_WIDTH` in [spec-fabric-error.md](./spec-fabric-error.md).

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
  A nonzero `res_tag` when writing a register is a configuration error
  (`CFG_TEMPORAL_PE_REG_TAG_NONZERO`). See [spec-fabric-error.md](./spec-fabric-error.md).

When a register has multiple readers, it behaves like an internal fork. The
FIFO entry is retained until each dependent instruction has fired once using
that entry. Only after all readers consume the entry is it dequeued.

## Error Conditions

The temporal PE raises a hardware error if any of the following occurs:

- An input tag matches no instruction (`RT_TEMPORAL_PE_NO_MATCH`).
- Duplicate instruction tags are configured (`CFG_TEMPORAL_PE_DUP_TAG`).
- An instruction specifies illegal register indices (`CFG_TEMPORAL_PE_ILLEGAL_REG`).
- A result writes to a register with `res_tag != 0` (`CFG_TEMPORAL_PE_REG_TAG_NONZERO`).

Errors are reported through a hardware-valid error signal and an error code
propagated to the top level. The corresponding symbols are
`RT_TEMPORAL_PE_NO_MATCH`, `CFG_TEMPORAL_PE_DUP_TAG`,
`CFG_TEMPORAL_PE_ILLEGAL_REG`, and `CFG_TEMPORAL_PE_REG_TAG_NONZERO`. See
[spec-fabric-error.md](./spec-fabric-error.md).

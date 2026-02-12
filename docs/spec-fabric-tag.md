# Fabric Tag Operations Specification

## Overview

The fabric dialect defines three tag boundary operations used to move between
native values and tagged values, or to transform tags across temporal regions:

- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

These operations operate on `!dataflow.tagged` values and are commonly used
at boundaries between dedicated and temporal regions.

## Operation: `fabric.add_tag`

### Syntax

```
%tagged = fabric.add_tag %value {tag = 5 : i4} : T -> !dataflow.tagged<T, i4>
```

### Operands

- `%value`: native value of type `T`.

### Results

- `%tagged`: tagged value of type `!dataflow.tagged<T, iN>`.

### Attributes

- `tag` (runtime configuration parameter)
  - Type: signless integer with the same width as the tag type.
  - Default value is `0`.
  - Must fit within the tag width.

### Constraints

- `T` must be a native value type.
- Tag type must satisfy `!dataflow.tagged` constraints from
  [spec-dataflow.md](./spec-dataflow.md). Tag-width violations raise
  `CPL_TAG_WIDTH_RANGE`.
- The result tagged type's value component must match the input type `T`.
  Violations raise `CPL_ADD_TAG_VALUE_TYPE_MISMATCH`.
- The configured `tag` value must fit in the result tag type width.
  Violations raise `CPL_ADD_TAG_VALUE_OVERFLOW`.

Example error:

```
// ERROR: CPL_ADD_TAG_VALUE_TYPE_MISMATCH
// Input is i32, but result value type is f32
%bad = fabric.add_tag %i32_val {tag = 0 : i4} : i32 -> !dataflow.tagged<f32, i4>
```

### Semantics

`fabric.add_tag` attaches a constant tag to the input value. In hardware, the
value and tag are concatenated into a single tagged payload. The tag is placed
in the high bits.

Timing model: combinational, zero-cycle data transformation.

## Operation: `fabric.del_tag`

### Syntax

```
%value = fabric.del_tag %tagged : !dataflow.tagged<T, iN> -> T
```

### Operands

- `%tagged`: tagged value of type `!dataflow.tagged<T, iN>`.

### Results

- `%value`: native value of type `T`.

### Constraints

- Output type must match the value type of the input tagged type.
  Violations raise `CPL_DEL_TAG_VALUE_TYPE_MISMATCH`.

Example error:

```
// ERROR: CPL_DEL_TAG_VALUE_TYPE_MISMATCH
// Input value type is i32, but output type is f32
%bad = fabric.del_tag %tagged : !dataflow.tagged<i32, i4> -> f32
```

### Semantics

`fabric.del_tag` removes the tag and forwards the value unchanged. The tag is
discarded.

Timing model: combinational, zero-cycle data transformation.

## Operation: `fabric.map_tag`

### Syntax

```
%retagged = fabric.map_tag %tagged
  {table_size = 10,
   table = [
     [1 : i1, 5 : i7, 2 : i3],
     [0 : i1, 0 : i7, 0 : i3],
     [1 : i1, 7 : i7, 3 : i3]
   ]}
  : !dataflow.tagged<T, i7> -> !dataflow.tagged<T, i3>
```

### Operands

- `%tagged`: tagged value of type `!dataflow.tagged<T, iM>`.

### Results

- `%retagged`: tagged value of type `!dataflow.tagged<T, iN>`.

### Attributes

- `table_size` (hardware parameter)
  - Unsigned integer.
  - Number of entries supported by the hardware table.
  - Must be at least 1 and at most 256.

- `table` (runtime configuration parameter)
  - Array of length `table_size`.
  - Each entry is a triple: `[valid, src_tag, dst_tag]`.
  - `valid` is `i1`.
  - `src_tag` uses the input tag type.
  - `dst_tag` uses the output tag type.
  - Entries with `valid = 0` are ignored.
  - For ignored entries, `src_tag` and `dst_tag` default to `0`.

### Machine Format (map_tag table entry)

For one `fabric.map_tag` table entry with input tag width `M` and output tag
width `N`, the packed bit layout is shown in **MSB -> LSB** order:

```
| dst_tag[N-1:0] | src_tag[M-1:0] | valid |
```

Bit-index rules:

- `valid` is always `entry[0]`.
- `src_tag` occupies `entry[M:1]`.
- `dst_tag` occupies `entry[M+N : M+1]`.

### Constraints

- Input and output value types must match. Violations raise
  `CPL_MAP_TAG_VALUE_TYPE_MISMATCH`.
- `table` length must equal `table_size`.
- Tag width can change between input and output.
- `valid` must be `i1`. Input and output tag types must satisfy
  `!dataflow.tagged` constraints from [spec-dataflow.md](./spec-dataflow.md).
  Tag-width violations raise `CPL_TAG_WIDTH_RANGE`.

Violations of table shape or size are compile-time errors: `CPL_MAP_TAG_TABLE_SIZE`
and `CPL_MAP_TAG_TABLE_LENGTH`. See [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`fabric.map_tag` transforms the tag using the runtime table while forwarding
its value unchanged.

Lookup rule:

- The table is searched for entries with `valid = 1` and `src_tag` equal to the
  input tag.
- Exactly one entry must match.
- If no entry matches, a runtime error is raised (`RT_MAP_TAG_NO_MATCH`).
- If multiple entries match, the table contains duplicate valid `src_tag`
  values and a configuration error is raised (`CFG_MAP_TAG_DUP_TAG`).
- The output tag is the `dst_tag` of the matching entry.

Errors are reported through a hardware-valid error signal and an error code
propagated to the top level. The corresponding symbols are
`RT_MAP_TAG_NO_MATCH` and `CFG_MAP_TAG_DUP_TAG`. See
[spec-fabric-error.md](./spec-fabric-error.md).

Timing model: combinational datapath with combinational table lookup.
When an error condition is detected, `error_valid`/`error_code` are captured
and held until reset.

## Related Documents

- [spec-fabric.md](./spec-fabric.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-fabric-error.md](./spec-fabric-error.md)

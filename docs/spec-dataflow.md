# Dataflow Dialect Specification

## Overview

The dataflow dialect provides small, explicit operations used to model
loop-carried stream semantics during lowering. The dialect currently defines
four operations and one type:

- `dataflow.carry`
- `dataflow.invariant`
- `dataflow.stream`
- `dataflow.gate`
- `!dataflow.tagged<value_type, tag_type>`

This document defines the syntax, constraints, and semantics of these elements.

## Common Type Conventions

This document uses the following terms:

- **Native value types**: scalar integers, scalar floats, and `index`.
- **Tagged type**: a native value type paired with a small integer tag.

The dataflow dialect does not define memref, tensor, vector, complex, or
opaque value types. Those are not valid as dataflow value types or as the value
component of `!dataflow.tagged`.

## Type: `!dataflow.tagged<value_type, tag_type>`

A tagged value pairs a native value with a small integer tag. The tag is
carried alongside the value and is treated as part of the data payload in
hardware.

### Syntax

```
!dataflow.tagged<value_type, tag_type>
```

Examples:

```
!dataflow.tagged<i32, i4>
!dataflow.tagged<f32, i5>
!dataflow.tagged<index, i1>
```

### Constraints

Allowed value types are listed below.

| Category | Allowed Types |
|----------|---------------|
| Integer | `i1`, `i8`, `i16`, `i32`, `i64` |
| Float | `bf16`, `f16`, `f32`, `f64` |
| Index | `index` |

- `tag_type` must be a signless integer type in the range `i1` to `i16`.
- `value_type` must be scalar. Vector, tensor, memref, complex, and opaque
  types are not allowed.

### Semantics

A tagged value is logically a pair `(value, tag)` where `tag` is an unsigned
integer of the given width. In hardware, the value and tag are treated as a
single packed word.

The packing convention is fixed:

- The value occupies the low bits.
- The tag occupies the high bits.

Example for `!dataflow.tagged<f32, i5>`:

- Bits `[31:0]` are the `f32` payload.
- Bits `[36:32]` are the `i5` tag.

## Operation: `dataflow.carry`

A loop-carried dependency state machine that aligns a value stream to a
control stream.

### Syntax

```
%o = dataflow.carry %d, %a, %b : i1, T, T -> T
```

### Operands

- `%d`: control stream, `i1`.
- `%a`: initial value stream, type `T`.
- `%b`: loop-carried value stream, type `T`.

### Results

- `%o`: output stream, type `T`.

### Constraints

- `%d` must be `i1`.
- `%a`, `%b`, and `%o` must have the same type.

### Semantics

`dataflow.carry` operates in two stages and repeats for each iteration burst.

1. Initial stage
2. Block stage

Initial stage behavior:

- Consumes one element from `%a`.
- Emits that element on `%o`, then switch to block stage.
- Does not consume `%d` in this stage.

Block stage behavior:

- Consumes elements of `%d` in order.
- For each `%d = true`, consumes one element from `%b` and emits it on `%o`.
- For each `%d = false`, consumes only `%d` and returns to the initial stage.

The length of `%o` matches the length of `%d` for each iteration burst.

Example:

```
%o = dataflow.carry %d, %a, %b : i1, i32, i32 -> i32

// Streams (T=true, F=false)
// a: [A, B]
// b: [C, D, E, F, G, H]
// d: [T, T, F, T, T, T, T, F]
// o: [A, C, D, B, E, F, G, H]
```

## Operation: `dataflow.invariant`

A loop-invariant value repeater aligned to a control stream.

### Syntax

```
%o = dataflow.invariant %d, %a : i1, T -> T
```

### Operands

- `%d`: control stream, `i1`.
- `%a`: invariant value stream, type `T`.

### Results

- `%o`: output stream, type `T`.

### Constraints

- `%d` must be `i1`.
- `%a` and `%o` must have the same type.

### Semantics

`dataflow.invariant` is similar to `dataflow.carry` but reuses the invariant
value instead of consuming a loop-carried stream.

Initial stage behavior:

- Consumes one element from `%a`.
- Emits that element on `%o`, and store that value internally, then switch to block stage.
- Does not consume `%d` in this stage.

Block stage behavior:

- Consumes elements of `%d` in order.
- For each `%d = true`, emits the stored value on `%o`.
- For each `%d = false`, consumes only `%d` and returns to the initial stage.

The length of `%o` matches the length of `%d` for each iteration burst.

Example:

```
%o = dataflow.invariant %d, %a : i1, i32 -> i32

// Streams (T=true, F=false)
// a: [A, B]
// d: [T, T, F, T, T, T, T, F]
// o: [A, A, A, B, B, B, B, B]
```

## Operation: `dataflow.stream`

An affine index stream generator for canonical `scf.for` loops.

### Syntax

```
%idx, %cont = dataflow.stream %start, %step, %bound
```

### Operands

- `%start`: `index`
- `%step`: `index`
- `%bound`: `index`

### Results

- `%idx`: `index` stream
- `%cont`: `i1` stream

### Constraints

- All operands must be `index`.
- `step` must be nonzero. If `step = 0` at runtime, the hardware raises
  `RT_DATAFLOW_STREAM_ZERO_STEP`. See [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`dataflow.stream` emits two streams for a loop with `N` iterations. The loop
direction is determined by the sign of `step`:

- If `step` is positive, the loop continues while `idx < bound`.
- If `step` is negative, the loop continues while `idx > bound`.

This supports both increasing and decreasing index streams.

`dataflow.stream` emits two streams for a loop with `N` iterations:

- `raw_index`: `start, start+step, ..., last, and one extra value`
- `raw_will_continue`: `true` for each body iteration, then `false` once

The streams have length `N + 1`. The extra element aligns with loop-carried
values that produce one more output than the body iteration count.

### State Machine Behavior

`dataflow.stream` is implemented as a two-phase state machine.

Initial phase:

- Wait for all three inputs: `%start`, `%step`, `%bound`.
- Output `idx = start`.
- Compute `willContinue`:
  - If `step > 0`, `willContinue = (start < bound)`.
  - If `step < 0`, `willContinue = (start > bound)`.
- Consume all three inputs.
- If `willContinue = true`, transition to block phase and latch:
  - `nextIdxReg = start + step`
  - `boundReg = bound`
  - `stepReg = step`
- If `willContinue = false`, remain in initial phase (zero-trip case).

Block phase:

- Output `idx = nextIdxReg`.
- Compute `willContinue`:
  - If `stepReg > 0`, `willContinue = (nextIdxReg < boundReg)`.
  - If `stepReg < 0`, `willContinue = (nextIdxReg > boundReg)`.
- If `willContinue = true`, update `nextIdxReg = nextIdxReg + stepReg` and stay
  in block phase.
- If `willContinue = false`, transition back to initial phase.

Example:

```
%idx, %cont = dataflow.stream %start, %step, %bound

// start=0, step=1, bound=5
// raw_index: [0, 1, 2, 3, 4, 5]
// raw_will_continue: [T, T, T, T, T, F]
```

Zero-trip example:

```
// start=3, step=1, bound=3
// raw_index: [3]
// raw_will_continue: [F]
```

## Operation: `dataflow.gate`

A stream adapter that aligns before-region and after-region loop streams.

### Syntax

```
%after_value, %after_cond = dataflow.gate %before_value, %before_cond
  : T, i1 -> T, i1
```

### Operands

- `%before_value`: stream of type `T`
- `%before_cond`: stream of type `i1`

### Results

- `%after_value`: stream of type `T`
- `%after_cond`: stream of type `i1`

### Constraints

- `%before_cond` must be `i1`.
- `%before_value` and `%after_value` must have the same type.
- `%after_cond` must be `i1`.

### Semantics

`dataflow.gate` performs a one-element shift to align the two streams:

- `after_value[i] = before_value[i]`
- `after_cond[i] = before_cond[i+1]`

This cuts the tail of `before_value` and the head of `before_cond`.
If `before_cond` has length 1, the outputs are empty.

Important timing detail:

- `after_cond[0]` corresponds to `after_value[1]`, not `after_value[0]`.
- This reflects that the last-iteration condition is only known after the
  next before-region condition is computed.

Example:

```
// before_value: [A, B, C, D, E]
// before_cond:  [T, T, T, T, F]
// after_value:  [A, B, C, D]
// after_cond:   [T, T, T, F]
```

## Change log and Reviews
- Date: 2026-02-03; Reviewer: [Sihao Liu](mailto:sihao@cs.ucla.edu); Note: initial draft.

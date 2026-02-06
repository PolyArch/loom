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
Compile-time errors for the software IR pipeline are listed in
[spec-dataflow-error.md](./spec-dataflow-error.md).

## Common Type Conventions

This document uses the following terms:

- **Native value types**: scalar integers, scalar floats, and `index`.
- **Tagged type**: a native value type paired with a small integer tag.

The dataflow dialect does not define memref, tensor, vector, complex, or
opaque value types. Those are not valid as dataflow value types or as the value
component of `!dataflow.tagged`.

## Control-Only Tokens

Loom represents control-only tokens (valid/ready without data payload) using
the MLIR `none` type. This applies to control tokens in `handshake.func`, as
well as load/store control and done tokens in the handshake/fabric memory
interfaces.

When a control-only token must carry a tag (for port multiplexing), it is
encoded as `!dataflow.tagged<i1, iK>`. The `i1` payload is a dummy value and
must be treated as constant `0` in hardware. It must not drive logic or carry
information. The tag is the only meaningful field and the `i1` payload is
expected to be optimized away in the backend.

For host-side ESI interfaces that cannot transport `none`, a control-only token
may be represented as an `i1` pulse with value `1`. This value must not drive
logic; it only indicates that a control token was issued. Hardware should treat
this as a one-cycle pulse which is consumed by the control path and then
cleared.

**Note on dataflow `i1` streams:** The `i1` control streams used by
`dataflow.carry`, `dataflow.invariant`, `dataflow.stream`, and `dataflow.gate`
are **boolean value streams**, not control-only tokens. Each element carries a
meaningful true/false value that controls loop behavior. These are distinct
from `none`-type control tokens which carry no data payload.

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

- `%d` must be `i1`. Violations raise `COMP_DATAFLOW_CARRY_CTRL_TYPE`.
- `%a`, `%b`, and `%o` must have the same type. Violations raise
  `COMP_DATAFLOW_CARRY_TYPE_MISMATCH`.

Example error:

```
// ERROR: COMP_DATAFLOW_CARRY_CTRL_TYPE
// %d is i32 but must be i1
%o = dataflow.carry %d, %a, %b : i32, i32, i32 -> i32
```

Example error:

```
// ERROR: COMP_DATAFLOW_CARRY_TYPE_MISMATCH
// %a is i32 but %b is f32
%o = dataflow.carry %d, %a, %b : i1, i32, f32 -> i32
```

See [spec-dataflow-error.md](./spec-dataflow-error.md).

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

- `%d` must be `i1`. Violations raise `COMP_DATAFLOW_INVARIANT_CTRL_TYPE`.
- `%a` and `%o` must have the same type. Violations raise
  `COMP_DATAFLOW_INVARIANT_TYPE_MISMATCH`.

Example error:

```
// ERROR: COMP_DATAFLOW_INVARIANT_CTRL_TYPE
// %d is i32 but must be i1
%o = dataflow.invariant %d, %a : i32, i32 -> i32
```

Example error:

```
// ERROR: COMP_DATAFLOW_INVARIANT_TYPE_MISMATCH
// %a is i32 but result %o is f32
%o = dataflow.invariant %d, %a : i1, i32 -> f32
```

See [spec-dataflow-error.md](./spec-dataflow-error.md).

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

A configurable index stream generator for loop-like control patterns.

### Syntax

```
%idx, %cont = dataflow.stream %start, %step, %bound
    {step_op = "+=", cont_cond = "<"}
```

### Operands

- `%start`: `index`
- `%step`: `index`
- `%bound`: `index`

### Results

- `%idx`: `index` stream
- `%cont`: `i1` stream

### Constraints

- All operands must be `index`. Violations raise
  `COMP_DATAFLOW_STREAM_OPERAND_TYPE`.
- `step_op` must be one of `+=`, `-=`, `*=`, `/=`, `<<=`, `>>=`. Invalid values
  raise `COMP_DATAFLOW_STREAM_INVALID_STEP_OP`.
- `cont_cond` must be one of `<`, `<=`, `>`, `>=`, `!=`. Invalid values raise
  `COMP_DATAFLOW_STREAM_INVALID_CONT_COND`.
- If `step_op` is omitted, it defaults to `+=`.
- If `cont_cond` is omitted, it defaults to `<`.
- `step` must be nonzero. If `step = 0` at runtime, the hardware raises
  `RT_DATAFLOW_STREAM_ZERO_STEP`. See [spec-fabric-error.md](./spec-fabric-error.md).

Example errors:

```
// ERROR: COMP_DATAFLOW_STREAM_OPERAND_TYPE
// %start is i32 but must be index
%idx, %cont = dataflow.stream %start, %step, %bound : i32, index, index

// ERROR: COMP_DATAFLOW_STREAM_INVALID_STEP_OP
// "%=" is not a valid step_op
%idx, %cont = dataflow.stream %start, %step, %bound {step_op = "%="}

// ERROR: COMP_DATAFLOW_STREAM_INVALID_CONT_COND
// "==" is not a valid cont_cond
%idx, %cont = dataflow.stream %start, %step, %bound {cont_cond = "=="}
```

See [spec-dataflow-error.md](./spec-dataflow-error.md).

`step_op` is a hardware parameter that determines the update operator and
cannot be changed at runtime. `cont_cond` is a runtime configuration parameter
that selects the comparison used by the loop controller.

In Fabric lowering, this runtime configuration is exposed only when a
`fabric.pe` body contains exactly one `dataflow.stream`. The PE contributes a
runtime `cont_cond_sel` field to `config_mem`. The authoritative encoding and
one-hot requirements are defined in
[spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md) and
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md). Configuration
violations use symbols from [spec-fabric-error.md](./spec-fabric-error.md).

### Semantics

`dataflow.stream` emits two streams for a loop with `N` iterations. It can
represent a canonical `scf.for`, or an analyzed `scf.while` that matches a
loop pattern but uses a non-`+=` update or a different comparison.

The continue condition is:

- `continue = (idx cont_cond bound)`

The next index update is:

- `next = idx (step_op) step`

- `raw_index`: `start, next, ..., last, and one extra value`
- `raw_will_continue`: `true` for each body iteration, then `false` once

The streams have length `N + 1`. The extra element aligns with loop-carried
values that produce one more output than the body iteration count.

**Extra value semantics:** The "extra value" is the index value that causes the
loop condition to fail. For example, with `start=0, step=1, bound=5, cont_cond="<"`,
the loop executes for indices 0, 1, 2, 3, 4 (5 iterations). The extra value is
5, which is the first index that fails the condition `5 < 5`. This extra value
is emitted alongside `willContinue = false` to signal loop termination.

### State Machine Behavior

`dataflow.stream` is implemented as a two-phase state machine.

Initial phase:

- Wait for all three inputs: `%start`, `%step`, `%bound`.
- Output `idx = start`.
- Compute `willContinue = (start cont_cond bound)`.
- Consume all three inputs.
- If `willContinue = true`, transition to block phase and latch:
  - `nextIdxReg = start (step_op) step`
  - `boundReg = bound`
  - `stepReg = step`
- If `willContinue = false`, remain in initial phase (zero-trip case).

Block phase:

- Output `idx = nextIdxReg`.
- Compute `willContinue = (nextIdxReg cont_cond boundReg)`.
- If `willContinue = true`, update `nextIdxReg = nextIdxReg (step_op) stepReg`
  and stay in block phase.
- If `willContinue = false`, transition back to initial phase.

Example:

```
%idx, %cont = dataflow.stream %start, %step, %bound
    {step_op = "+=", cont_cond = "<"}

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

Example: right shift with "!="

```
%idx, %cont = dataflow.stream %start, %step, %bound
    {step_op = ">>=", cont_cond = "!="}

// start=16, step=1, bound=1
// raw_index: [16, 8, 4, 2, 1]
// raw_will_continue: [T, T, T, T, F]
```

Example: left shift with "<="

```
%idx, %cont = dataflow.stream %start, %step, %bound
    {step_op = "<<=", cont_cond = "<="}

// start=1, step=1, bound=8
// raw_index: [1, 2, 4, 8, 16]
// raw_will_continue: [T, T, T, T, F]
```

## Handshake Memory Control Constraints

Scope note: this section describes a lowering-boundary constraint between
Dataflow usage and CIRCT Handshake operations. It remains in this document
because it directly constrains legal Dataflow-to-Handshake lowering behavior.
`handshake.func`, `handshake.load`, `handshake.store`, `handshake.memory`, and
`handshake.extmemory` are operations from the CIRCT Handshake dialect.

When lowering to `handshake.func`, the compiler constructs control-token chains
for each `handshake.load` and `handshake.store`. Each control token must be
rooted at the `handshake.func` `start_token`, or depend only on done tokens
produced by the same `handshake.extmemory` or `handshake.memory` associated with
that access. If a control chain depends on done tokens from other memory
interfaces, the compiler raises `COMP_HANDSHAKE_CTRL_MULTI_MEM`. See
[spec-dataflow-error.md](./spec-dataflow-error.md).

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

- `%before_cond` must be `i1`. Violations raise `COMP_DATAFLOW_GATE_COND_TYPE`.
- `%before_value` and `%after_value` must have the same type. Violations raise
  `COMP_DATAFLOW_GATE_TYPE_MISMATCH`.
- `%after_cond` must be `i1`. Violations raise `COMP_DATAFLOW_GATE_COND_TYPE`.

Example error:

```
// ERROR: COMP_DATAFLOW_GATE_COND_TYPE
// %before_cond is i32 but must be i1
%after_value, %after_cond = dataflow.gate %before_value, %before_cond
  : i32, i32 -> i32, i1

// ERROR: COMP_DATAFLOW_GATE_TYPE_MISMATCH
// %before_value is i32 but %after_value is f32
%after_value, %after_cond = dataflow.gate %before_value, %before_cond
  : i32, i1 -> f32, i1
```

See [spec-dataflow-error.md](./spec-dataflow-error.md).

### Semantics

`dataflow.gate` performs a one-element shift to align the two streams:

- `after_value[i] = before_value[i]`
- `after_cond[i] = before_cond[i+1]`

This cuts the tail of `before_value` and the head of `before_cond`.
If `before_cond` has length 1, the outputs are empty.

Important timing detail:

- `after_cond[0]` gates whether `after_value[1]` will be produced, not
  `after_value[0]`.
- This reflects that the continue condition for iteration `i` is known only
  after the before-region of iteration `i+1` is evaluated.

Example:

```
// before_value: [A, B, C, D, E]
// before_cond:  [T, T, T, T, F]
// after_value:  [A, B, C, D]
// after_cond:   [T, T, T, F]
```

## Related Documents

- [spec-fabric-pe.md](./spec-fabric-pe.md): Fabric PE specification (dataflow ops inside PEs)
- [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md): Allowed operations in fabric.pe (includes dataflow ops)
- [spec-fabric-mem.md](./spec-fabric-mem.md): Memory operations using tagged/native types
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md): Runtime configuration layout used by dataflow.stream lowering
- [spec-fabric-error.md](./spec-fabric-error.md): Fabric CFG_/RT_ symbols referenced by embedded dataflow behavior
- [spec-fabric.md](./spec-fabric.md): Fabric dialect overview and type conventions
- [spec-dataflow-error.md](./spec-dataflow-error.md): Dataflow error code definitions

## Change log and Reviews
- Date: 2026-02-03; Reviewer: [Sihao Liu](mailto:sihao@cs.ucla.edu); Note: initial draft.

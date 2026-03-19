# FCC Dataflow Dialect Specification

## Overview

FCC uses a small `dataflow` dialect to represent loop-derived stream and
state-machine behavior in DFG form.

Current core operations are:

- `dataflow.stream`
- `dataflow.gate`
- `dataflow.carry`
- `dataflow.invariant`

These are software-graph operations. When they are implemented in hardware,
FCC treats each of them as one dedicated fixed-behavior state machine.

Related documents:

- [spec-compilation.md](./spec-compilation.md)
- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)

## Relationship to Fabric Types

The `dataflow` dialect itself does not define a dedicated tagged carrier type.

Instead:

- software-side `dataflow` operations use native semantic values such as
  `index`, `i1`, or other scalar element types
- when a mapped hardware path carries those values across ADG boundaries, FCC
  uses Fabric transport types such as `!fabric.bits<...>` or
  `!fabric.tagged<...>`

Therefore the separation is:

- `dataflow.*` defines the software-visible stream semantics
- `fabric.bits` and `fabric.tagged` define the hardware transport container

## Common Timing Rule for Hardware Mapping

All four current `dataflow` operations are fixed state-machine behaviors.

Normative FCC rule for hardware mapping:

- a `function_unit` that implements any one of the four `dataflow` operations
  must contain exactly that one non-terminator `dataflow` operation
- such a `function_unit` must use `latency = -1` and `interval = -1`

FCC does not currently model any of the four `dataflow` operations as ordinary
single-fire datapath FUs with meaningful scalar `latency` or `interval`.

The authoritative FU-body legality, timing, and exclusivity rules are defined
in [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md).

## Operation: `dataflow.stream`

`dataflow.stream` is a configurable index-stream generator.

### Syntax

```mlir
%idx, %cont = dataflow.stream %start, %step, %bound
    {step_op = "+=", cont_cond = "<"}
    : (index, index, index) -> (index, i1)
```

### Operands

- `%start`: `index`
- `%step`: `index`
- `%bound`: `index`

### Results

- `%idx`: generated index stream
- `%cont`: boolean will-continue stream, type `i1`

### Attributes

- `step_op`: update operator
- `cont_cond`: loop-continuation comparison

Current valid `step_op` values are:

- `+=`
- `-=`
- `*=`
- `/=`
- `<<=`
- `>>=`

Current valid `cont_cond` values are:

- `<`
- `<=`
- `>`
- `>=`
- `!=`

### Semantics

`dataflow.stream` operates as a two-phase generator.

Phase A: activation

- wait until `%start`, `%step`, and `%bound` are all available
- latch these three values internally
- mark the stream generator active

Phase B: emit one `(idx, cont)` pair per firing

- evaluate `cont = (next_idx cont_cond bound)`
- emit the current `next_idx` on `%idx`
- emit the boolean result on `%cont`
- if `cont = true`, update `next_idx = next_idx step_op step`
- if `cont = false`, terminate the current activation and return to Phase A

### Stream Shape

For a loop with `N` body iterations:

- `%idx` has `N + 1` elements
- `%cont` has `N` true values followed by one false value

This is the canonical FCC representation of the one-step-ahead loop-control
stream used by later `dataflow.gate`, `dataflow.carry`, and
`dataflow.invariant`.

Example:

- `start = 0`
- `step = 1`
- `bound = 4`
- `step_op = +=`
- `cont_cond = <`

Then:

- `%idx = [0, 1, 2, 3, 4]`
- `%cont = [true, true, true, true, false]`

### Hardware Mapping Note

The common dedicated-dataflow FU rule above applies.

Additional mapping note:

- `cont_cond` contributes runtime configuration bits
- `step_op` is structural, not runtime-configurable

## Operation: `dataflow.gate`

`dataflow.gate` shifts a one-step-ahead before-region stream into a body-region
stream.

### Syntax

```mlir
%after_value, %after_cond = dataflow.gate %before_value, %before_cond
    : T, i1 -> T, i1
```

### Operands

- `%before_value`: before-region value stream, type `T`
- `%before_cond`: before-region loop-condition stream, type `i1`

### Results

- `%after_value`: body-region value stream, type `T`
- `%after_cond`: body-region loop-condition stream, type `i1`

### Semantics

`dataflow.gate` is the alignment adapter between:

- the before-region stream shape produced by `dataflow.stream` or
  `dataflow.carry`
- the body-region stream shape consumed by loop-body operations

Its behavior is a fixed two-phase state machine.

#### Phase 1: `NeedHead`

In this phase, `dataflow.gate` consumes one `(before_value, before_cond)` pair
from its inputs.

If `before_cond = true`:

- consume both inputs
- emit only `after_value = before_value`
- transition to Phase 2

If `before_cond = false`:

- consume both inputs
- emit nothing
- remain in Phase 1

This phase discards the final tail element associated with the terminating
`false` condition when no new loop body should start.

#### Phase 2: `NeedNext`

In this phase, `dataflow.gate` again consumes one
`(before_value, before_cond)` pair.

If `before_cond = true`:

- consume both inputs
- emit `after_value = before_value`
- emit `after_cond = true`
- remain in Phase 2

If `before_cond = false`:

- consume both inputs
- emit only `after_cond = false`
- transition back to Phase 1

#### Net Effect

`dataflow.gate` transforms an `N + 1` before-region stream pair into an `N`
body-region stream pair:

- one initial `true` launches the body stream
- all interior `true` values produce `(after_value, true)`
- the terminating `false` produces only `after_cond = false`

This is why `dataflow.gate` is not equivalent to `dataflow.invariant` and not
equivalent to a simple combinational filter.

Example:

- `%before_value = [0, 1, 2, 3, 4]`
- `%before_cond = [true, true, true, true, false]`

Then:

- `%after_value = [0, 1, 2, 3]`
- `%after_cond = [true, true, true, false]`

### Hardware Mapping Note

The common dedicated-dataflow FU rule above applies.

## Operation: `dataflow.carry`

`dataflow.carry` models one loop-carried dependency state machine.

### Syntax

```mlir
%o = dataflow.carry %d, %a, %b : i1, T, T -> T
```

### Operands

- `%d`: loop-condition stream, type `i1`
- `%a`: initial value stream, type `T`
- `%b`: loop-carried value stream, type `T`

### Results

- `%o`: carried output stream, type `T`

### Semantics

`dataflow.carry` alternates between an initial stage and a loop stage.

#### Phase 1: `NeedInit`

- wait for one token on `%a`
- consume `%a`
- emit that value on `%o`
- transition to Phase 2

#### Phase 2: `NeedCond`

- wait for one token on `%d`
- consume `%d`

If `%d = true`:

- transition to Phase 3

If `%d = false`:

- transition back to Phase 1

#### Phase 3: `NeedLoop`

- wait for one token on `%b`
- consume `%b`
- emit that value on `%o`
- transition back to Phase 2

This models the classic loop-carried recurrence:

- emit one initial value
- then emit one carried value for each continuing iteration
- reset when the loop terminates

### Hardware Mapping Note

The common dedicated-dataflow FU rule above applies.

## Operation: `dataflow.invariant`

`dataflow.invariant` models one loop-invariant value repeater.

### Syntax

```mlir
%o = dataflow.invariant %d, %a : i1, T -> T
```

### Operands

- `%d`: loop-condition stream, type `i1`
- `%a`: invariant seed stream, type `T`

### Results

- `%o`: invariant replay stream, type `T`

### Semantics

`dataflow.invariant` stores one invariant value and replays it while the loop
condition remains true.

#### Phase 1: `NeedInit`

- wait for one token on `%a`
- consume `%a`
- store it internally
- emit it once on `%o`
- transition to Phase 2

#### Phase 2: `NeedCond`

- wait for one token on `%d`
- consume `%d`

If `%d = true`:

- emit the stored invariant value on `%o`
- remain in Phase 2

If `%d = false`:

- emit nothing
- transition back to Phase 1

This means:

- the initial invariant value is consumed once per iteration burst
- that value is replayed for the active iterations of that burst
- a terminating `false` resets the machine for the next burst

### Hardware Mapping Note

The common dedicated-dataflow FU rule above applies.

## Mapper Contract

For FCC mapping:

- each of the four `dataflow` operations is a distinct compatibility class
- matching is by exact operation identity
- no implicit equivalence exists between `stream`, `gate`, `carry`, and
  `invariant`
- a hardware FU that implements one of the four `dataflow` operations must not
  be reused as a generic compound FU for other non-dataflow operations

In particular:

- `dataflow.invariant` must not be matched through `dataflow.gate`
- `dataflow.gate` must not be matched through `dataflow.invariant`

## Relationship to Loop Lowering

FCC's SCF-to-DFG lowering uses the four operations in a specific pattern:

- `dataflow.stream` generates one-step-ahead loop index and continue streams
- `dataflow.gate` converts before-region shape into body-region shape
- `dataflow.carry` models loop-carried values
- `dataflow.invariant` models values reused across loop iterations

This is why `dataflow.gate` is timing-critical in the model: it is the
alignment boundary between the before-loop stream domain and the loop-body
stream domain.

## Visualization and Simulation Contract

Focused tests should exist for:

- standalone `dataflow.stream`
- standalone `dataflow.gate`
- standalone `dataflow.carry`
- standalone `dataflow.invariant`
- at least one representative `stream -> gate -> carry` chain

These tests should validate:

- token ordering
- stage transitions
- final reset behavior after one terminating `false`
- mapping correctness when one dataflow op is implemented by one dedicated FU

## Related Documents

- [spec-compilation.md](./spec-compilation.md)
- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)

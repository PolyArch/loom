# FCC Dataflow Compilation Specification

## Overview

This document specifies how FCC lowers nested `scf` structure into
`handshake + dataflow` DFG form.

The focus here is not the standalone semantics of the four `dataflow`
operations. Those are defined in [spec-dataflow.md](./spec-dataflow.md).

The focus here is:

- how `scf.for`, `scf.while`, and `scf.if` are flattened
- which control stream is `raw`
- which control stream is body-visible
- where `dataflow.stream`, `dataflow.gate`, `dataflow.carry`, and
  `dataflow.invariant` must be inserted
- how loop-carried values, exit values, and nested-region values are wired

This is one of the most delicate parts of the FCC compilation pipeline.
The normative intent follows the older
`dsa-stack/main/lib/dsa/Transforms/SCFToHandshakeDSA` implementation, which
had already been checked by simulation.

Related documents:

- [spec-dataflow.md](./spec-dataflow.md)
- [spec-dataflow-memory.md](./spec-dataflow-memory.md)
- [spec-compilation.md](./spec-compilation.md)

## Terms

### Raw Loop Control

For a stream-derived loop, the value produced directly by
`dataflow.stream.willContinue` is the raw loop-control stream.

Properties:

- length is `N + 1` for a loop body that executes `N` times
- it corresponds to the before-region condition stream
- it contains the final terminating `false`

This document names that stream `raw_will_continue`.

### Body-Visible Loop Control

The value produced by `dataflow.gate.afterCond` is the body-visible
loop-control stream.

Properties:

- length is `N`
- it is the after-region or body-region control stream
- it still ends with one final `false`
- all body-local execution control must use this stream unless a rule below
  explicitly requires the raw stream

This document names that stream `after_cond`.

### SCF Block Abstraction

During lowering, every SCF region is treated as an explicit dataflow block
with:

- a value environment
- an execution-control stream
- zero or more exit values

The lowering must turn all implicit SCF structure into explicit SSA dataflow
edges.

## Normative Rule: Raw vs Gated Control

FCC must distinguish two different uses of loop control.

### Uses that must take `raw_will_continue`

- loop-carried bridge `dataflow.carry` that models `scf.for iter_args`
- split of an `iter_arg` carry output into:
  - body-visible true branch
  - loop-exit false branch
- any value that semantically belongs to the before-region or loop-boundary
  interface rather than the loop body

### Uses that must take `after_cond`

- body-local `dataflow.invariant`
- body-local control token replication
- memory ordering and memory control inside the loop body
- any loop-body-only control consumer
- recursive lowering of nested SCF that resides inside the loop body

This distinction is mandatory.

Using `raw_will_continue` for body-local control is incorrect because it
reintroduces the extra `(N + 1)`th control event that belongs to the
before-region only.

## Lowering of `scf.if`

`scf.if` is flattened into explicit branch and merge operators.

### Structural Form

For a condition `%cond`:

- branch entry control by `handshake.cond_br`
- lower then-region recursively
- lower else-region recursively
- branch-gate any yielded value that is defined outside the corresponding
  branch region, so each branch result is locally owned before merge
- merge branch results with `handshake.mux`

### Reference Pseudocode

```text
function lower_if(if_op, parent_state):
  cond = map_value(if_op.condition, parent_state)
  ctrl_in = parent_state.control

  then_ctrl, else_ctrl = cond_br(cond, ctrl_in)

  then_state = child_state(parent_state)
  then_state.control = then_ctrl
  then_results, then_done = lower_region(if_op.then_region, then_state)

  else_state = child_state(parent_state)
  else_state.control = else_ctrl
  else_results, else_done = lower_region(if_op.else_region, else_state)

  merged_results = []
  for each pair (then_v, else_v):
    if then_v is defined outside then_region:
      then_v = cond_br(cond, then_v).true_result
    if else_v is defined outside else_region:
      else_v = cond_br(cond, else_v).false_result
    merged_results.push(mux(sel = cond, false = else_v, true = then_v))

  merged_done = mux(sel = cond, false = else_done, true = then_done)
  return merged_results, merged_done
```

### Reference Block Diagram

```text
                +------------------- then region -------------------+
                |                                                   |
parent_ctrl ---> cond_br(cond, ctrl) ---- true ----> [ lower then ] --- then_done
                |                                                   |
                |                                                   |
                +---- false ----> [ lower else ] --- else_done -----+
                                   |                    |
                                   +---- values --------+

result_i = mux(cond, else_value_i, then_value_i)
done     = mux(cond, else_done,    then_done)
```

### Rules

- the branch condition is the mapped SSA condition, not a synthetic loop token
- values used from outside a branch region must be replicated into the branch
  with `dataflow.invariant` if they are body-local values
- if a yielded value is defined outside the branch region, it must first be
  split by `handshake.cond_br` so the untaken branch does not leave a residual
  token on the final `handshake.mux`
- branch-local side effects must stay inside that branch subtree
- branch outputs must be merged explicitly; they do not implicitly rejoin

### Zero-Trip Guard

When `scf.if` is only a canonical zero-trip guard around a loop, the resulting
DFG still follows the same branch rule. The enclosing loop lowering must not
assume the guard disappears semantically.

## Lowering of `scf.for`

### Base Shape

An `scf.for` loop is lowered to:

1. one `dataflow.stream`
2. one `dataflow.gate`
3. zero or more `dataflow.carry` for `iter_args`
4. zero or more `dataflow.invariant` for values defined outside the loop body
5. recursively lowered body operations

### Stream and Gate

For:

```text
scf.for %iv = %lb to %ub step %step
```

the lowering must create:

- `raw_index, raw_will_continue = dataflow.stream(lb, step, ub, ...)`
- `body_index, after_cond = dataflow.gate(raw_index, raw_will_continue)`

Then:

- `%iv` maps to `body_index`
- loop-body execution control maps to `after_cond`

### Reference Pseudocode

```text
function lower_for(for_op, parent_state):
  lb   = map_value(for_op.lb,   parent_state)
  ub   = map_value(for_op.ub,   parent_state)
  step = map_value(for_op.step, parent_state)

  raw_index, raw_will_continue = stream(lb, step, ub)
  body_index, after_cond = gate(raw_index, raw_will_continue)

  body_state = child_state(parent_state)
  body_state.control = invariant(after_cond, parent_state.control)
  body_state.iv = body_index

  for each external body-local value ext_v:
    body_state.map(ext_v) = invariant(after_cond, map_value(ext_v, parent_state))

  for each iter_arg (init_i):
    carry_o_i = carry(raw_will_continue, init_i, feedback_i_placeholder)
    body_iter_i, exit_iter_i = split_by_raw_cond(raw_will_continue, carry_o_i)
    body_state.map(iter_arg_i) = body_iter_i
    parent_state.map(loop_result_i) = exit_iter_i

  body_results, body_done = lower_region(for_op.body, body_state)

  for each iter_arg i:
    patch carry_i.feedback = body_results.yield_i

  return loop_results, body_done
```

### Reference Block Diagram: `scf.for` Without `iter_args`

```text
lb, step, ub
   |
   v
[ dataflow.stream ] ---- raw_index -----------+
        |                                     |
        +---- raw_will_continue ----+         |
                                    v         v
                           [ dataflow.gate(raw_index, raw_will_continue) ]
                                    |                    |
                                    |                    +--> after_cond
                                    +--> body_index

parent_ctrl -- invariant(after_cond, parent_ctrl) --> body_ctrl

body arguments:
  iv   <- body_index
  ctrl <- body_ctrl
  ext  <- invariant(after_cond, ext)

body-local values and memory-control use after_cond only
```

### Loop Invariants

If a value is defined outside the loop but used inside the loop body, and the
value is not a memref transport object, it must be replicated with
`dataflow.invariant`.

The controlling operand of that invariant must be `after_cond`, not
`raw_will_continue`.

Reason:

- the invariant is a body-local value stream
- its length must match body execution count `N`
- it must not receive the extra before-region control event

### `iter_args`

Each `scf.for iter_arg` is lowered as a loop-boundary `dataflow.carry`.

For one iter arg:

```text
iter_arg init = %a0
yield feedback = %b
```

the normative structure is:

1. build `%carry_o = dataflow.carry(raw_will_continue, %a0, %b)`
2. split `%carry_o` by `raw_will_continue`
3. route:
   - true branch to the body block argument
   - false branch to the loop result

Conceptually:

```text
carry_o = carry(raw_will_continue, init, feedback)
body_value, exit_value = split_by_raw_cond(raw_will_continue, carry_o)
```

The split may be realized with `handshake.cond_br`, or with an equivalent
construction that preserves the same stream lengths and exit semantics.

### Reference Pseudocode: `scf.for` With `iter_args`

```text
function lower_for_iter_arg(init_i, raw_will_continue, body_state):
  carry_o = carry(raw_will_continue, init_i, feedback_placeholder)
  body_iter, exit_iter = split_by_raw_cond(raw_will_continue, carry_o)

  body_state.map(iter_arg_i) = body_iter
  loop_result_i = exit_iter

  return carry_o, loop_result_i
```

### Reference Block Diagram: `scf.for` With `iter_args`

```text
                         +------------------------------+
raw_will_continue ------>| dataflow.carry(raw, init, feedback) |---- carry_o
                         +------------------------------+
                                                      |
                                                      v
                                 split_by_raw_cond(raw_will_continue, carry_o)
                                         |                         |
                                         | true                    | false
                                         v                         v
                                   body iter_arg               loop result

body uses:
  iv            <- body_index
  iter_arg      <- true branch of carry split
  invariants    <- after_cond
  memory ctrl   <- after_cond
```

### Why `iter_args` use `raw_will_continue`

The loop-boundary carry belongs to the interface between:

- the before-region view of the loop
- the after-region or body view of the loop
- the loop exit

It must therefore produce the `N + 1` sequence:

- one initial value
- `N - 1` body feedback values
- one final exit value

This is different from a body-local invariant or memory-control carry, which
must only observe the `N` body-visible control events.

### Loops Without `iter_args`

Even when `scf.for` has no results, the split between raw control and
body-visible control still exists.

In particular:

- index generation still uses `stream + gate`
- body-local control still uses `after_cond`
- memory ordering inside the body still uses `after_cond`

### Reference Block Diagram: Control Split

```text
raw_will_continue  : before-region control, length N+1
after_cond         : body-visible control,  length N

raw_will_continue --> iter_arg carries and loop-boundary exit split
after_cond        --> invariants, body control, nested-SCF body lowering,
                      memory-control, and all body-local consumers
```

## Lowering of `scf.while`

FCC lowers `scf.while` using the same four dataflow primitives, but with an
explicit before-region / after-region partition.

### Region Meaning

- before-region computes the next loop condition and trailing args
- after-region is the loop body that runs only when the condition is true

### Normative Structure

For each while-carried value:

- build a `dataflow.carry` that feeds the before-region
- compute the raw condition in the before-region
- gate the raw condition into `after_cond`
- route condition args through explicit true/false split
  - true branch enters the after-region
  - false branch becomes the loop exit value
- use after-region yield values as carry feedback

### Reference Pseudocode

```text
function lower_while(while_op, parent_state):
  for each operand init_i:
    carry_i = carry(raw_cond_placeholder, init_i, feedback_i_placeholder)

  before_state = child_state(parent_state)
  before_state.control = invariant(raw_cond_placeholder, parent_state.control)
  before_state.map(before_arg_i) = carry_i.output

  cond_value, cond_args = lower_before_region(while_op.before, before_state)
  raw_will_continue = cond_value

  patch all carries.ctrl = raw_will_continue
  patch before invariants.ctrl = raw_will_continue

  dummy, after_cond = gate(dummy_value_repeated_by(raw_will_continue),
                           raw_will_continue)

  for each cond_arg_i:
    after_arg_i, exit_arg_i = split_by_raw_cond(raw_will_continue, cond_arg_i)

  after_state = child_state(parent_state)
  after_state.control = invariant(after_cond, before_state.control)
  after_state.map(after_region_arg_i) = after_arg_i

  yield_values, after_done = lower_after_region(while_op.after, after_state)

  for each carry_i:
    patch carry_i.feedback = yield_value_i

  return exit_args, after_done
```

### Reference Block Diagram

```text
 init_i --> carry_i --+--> before region --------------------+
                      |                                       |
                      +------> raw condition -----------------+--> raw_will_continue
                                                              |
                                                              v
                                                     [ gate to after_cond ]
                                                              |
                                    +-------------------------+------------------+
                                    |                                            |
                                    v                                            v
                        split condition args by raw cond                  invariant(after_cond, ...)
                                    | true                                       |
                                    v                                            v
                               after region ------------------------------> yield feedback
                                    |
                                    +--> false branch = while exit values
```

### Rule

For `scf.while`, just as for `scf.for`:

- before-region interfaces use the raw condition stream
- after-region body execution uses the gated condition stream

## Nested SCF Flattening

FCC must flatten nested SCF recursively.

### General Rule

When lowering an SCF region:

- clone non-SCF operations into the current handshake/dataflow region
- recursively lower nested SCF operations in place
- replace region-local implicit control with explicit dataflow edges

### Reference Pseudocode

```text
function lower_region(region, region_state):
  for each op in region in source order:
    if op is scf.if:
      lower_if(op, region_state)
    else if op is scf.for:
      lower_for(op, region_state)
    else if op is scf.while:
      lower_while(op, region_state)
    else if op is terminator:
      collect mapped terminator operands
    else:
      clone op with mapped operands
  return region_results, region_done
```

### Value Mapping Rule

When a nested region uses a value from an outer region:

- if the value is body-local to the enclosing loop body, replicate it with
  `dataflow.invariant(after_cond, value)`
- if the value is a loop-carried boundary value, use the corresponding carry
  true branch
- if the value is a loop-exit value, use the false branch of the boundary
  split

### No Implicit Region Inheritance

After lowering, there must be no semantic dependence on original SCF nesting.
All dependencies must appear as explicit SSA edges.

### Reference Block Diagram: Nested `scf.for` Body Containing `scf.if`

```text
outer stream/gate --> outer after_cond --> body invariants/body control
                                      |
                                      +--> lower nested scf.if
                                               |
                                               +--> cond_br
                                               +--> lower then subtree
                                               +--> lower else subtree
                                               +--> mux merge

No nested region inherits control implicitly.
Every branch and loop receives an explicit mapped control stream.
```

## Worked Example: `scf.for` Without `iter_args`

Source shape:

```text
for i = lb .. ub step step {
  body(i)
}
```

Lowered shape:

```text
raw_index, raw_will_continue = stream(lb, step, ub)
body_index, after_cond = gate(raw_index, raw_will_continue)
body_ctrl = invariant(after_cond, parent_ctrl)
body(body_index, after_cond, body_ctrl, ...)
```

Key point:

- body-local operations never consume `raw_will_continue`

## Worked Example: `scf.for` With `iter_args`

Source shape:

```text
%r = scf.for ... iter_args(%x = %init) {
  ...
  scf.yield %next
}
```

Lowered shape:

```text
raw_index, raw_will_continue = stream(...)
body_index, after_cond = gate(raw_index, raw_will_continue)

carry_o = carry(raw_will_continue, init, next_feedback)
body_x, exit_x = split_by_raw_cond(raw_will_continue, carry_o)

body uses:
  iv = body_index
  iter_arg = body_x
  invariants and memory control use after_cond

loop result = exit_x
```

Key point:

- loop-carried values are special boundary values
- body-local consumers still use the gated stream

## Worked Example: `scf.for` With Body Memory Control

Source shape:

```text
scf.for ... {
  %x = memref.load ...
  memref.store %x, ...
}
```

Lowered shape:

```text
raw_index, raw_will_continue = stream(...)
body_index, after_cond = gate(raw_index, raw_will_continue)

body memory data path:
  load[address = body_index]
  store[address = body_index, data = ...]

body memory control path:
  loop_ctrl = carry(after_cond, entry_ctrl, body_done_feedback)
  body_done_true, loop_exit_done = split_by_after_cond(after_cond, body_done)

Important:
  memory control uses after_cond
  never raw_will_continue
```

## Compilation Invariants

The following must always hold after lowering.

### Length Invariants

- `dataflow.stream` emits `N + 1`
- `dataflow.gate` reduces before-region streams to `N`
- body-local invariants are `N`
- `iter_arg` boundary carries are `N + 1`
- loop-body memory control is `N`

### Control Attachment Invariants

- `iter_arg carry.d` uses `raw_will_continue`
- body-local `invariant.d` uses `after_cond`
- loop-body memory-control carries use `after_cond`
- any loop-body-only `cond_br` that represents body control uses `after_cond`

### Exit Invariants

- loop exit values come from the false branch of a boundary split
- body-local consumers must never observe the extra terminal raw event

## Non-Normative Debugging Guidance

When a mapped loop finishes with:

- one extra control token in `load/store ctrl`
- one extra control token in `cond_br cond`
- data-path counts correct but control-path counts larger by one

the first thing to check is whether a body-local control chain was wired to
`raw_will_continue` instead of `after_cond`.

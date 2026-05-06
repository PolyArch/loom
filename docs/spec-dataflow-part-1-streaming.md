# Loom Dataflow Part 1: Streaming Ops

This document specifies the precise token timing semantics of the
stream-shaping ops in the `dataflow` dialect:
`dataflow.stream`, `dataflow.carry`, `dataflow.invariant`, and
`dataflow.gate`.

The canonical IR source is `include/Dataflow/IR/DataflowOps.td`; the
verifier implementation lives in `lib/Dataflow/IR/DataflowOps.cpp`.
This document is the design-level companion for compiler specs that use
these ops, especially `docs/spec-compiler-part-3-dfg.md`.

## 1. Token Stream Model

A dataflow SSA value denotes an ordered token stream. A token may carry
ordinary data, such as an integer, floating point value, vector, or
`none`, or it may represent a control event. Multiple SSA uses of the
same stream are token broadcast: each use observes the same ordered
token sequence.

The four ops in this document shape streams. They do not define memory
regions. A `memref<...>` value in the compiler frontend represents a
memory-region binding. It must not be duplicated, delayed, or
phase-normalized by `dataflow.carry`, `dataflow.invariant`, or
`dataflow.gate`. Dynamic control chooses address, data, operation, and
`none` ordering streams; it does not choose the frontend memref binding
itself. Memory accesses use the memref as the static binding operand of
`dataflow.load` / `dataflow.store`; ordering is carried by explicit
`none` tokens.

## 2. RWC Streams

An `rwc : i1` stream is a run/continue stream for a repeated region.
It is not a plain validity bit. For a region that executes `N` times,
the canonical loop-level stream has `N + 1` tokens:

* `N` true tokens, one per dynamic execution.
* One trailing false token, the sentinel that closes or resets
  stateful stream-shaping ops.

The trailing false token is semantically observable. Lowering must not
drop it before it reaches every stateful op that needs to close or
reset. Values that should be consumed only by the repeated body must be
converted into body phase, normally with `dataflow.gate`.

## 3. `dataflow.stream`

`dataflow.stream` produces an index stream plus a loop-level `rwc`
stream from scalar integer bounds:

```
%index, %rwc = dataflow.stream %lb, %ub, %step
  {step_op = "+=", cont_cond = "<"} : iN
```

The operation keeps a current index initialized to `%lb`.

* If `cont_cond(current, ub)` is true, it emits
  `(current, true)`, advances the current index with `step_op`, and
  repeats.
* If `cont_cond(current, ub)` is false, it emits one final
  `(current, false)` pair and stops.

The current dialect definition requires `%lb`, `%ub`, `%step`, and
`%index` to share a signless integer-like type. The `rwc` result is
always `i1`.

For `lb = 0`, `ub = 5`, `step = 1`, `step_op = "+="`, and
`cont_cond = "<"`, the output is:

| Token | `%index` | `%rwc` |
|-------|----------|--------|
| 0 | 0 | true |
| 1 | 1 | true |
| 2 | 2 | true |
| 3 | 3 | true |
| 4 | 4 | true |
| 5 | 5 | false |

For a zero-trip counted loop such as `lb = 5`, `ub = 5`, `step = 1`,
and `cont_cond = "<"`, the output is exactly one sentinel pair:

| Token | `%index` | `%rwc` |
|-------|----------|--------|
| 0 | 5 | false |

The sentinel index is not a body induction value. Body address
computation, memory effects, and ordinary body operations must consume
the gated body-phase index, not the raw stream index.

## 4. `dataflow.carry`

`dataflow.carry` is a two-state token element for loop-carried values
or hidden loop-carried `none` state:

```
%output = dataflow.carry %cond, %init, %carry : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, forwards it to
  `%output`, and transitions to `carry` state.
* In `carry` state, it waits for one `%cond : i1` token and one
  `%carry` token simultaneously, then consumes both.
* If `%cond` is true, it forwards `%carry` to `%output` and stays in
  `carry` state.
* If `%cond` is false, it emits no output and returns to `init` state.

`%init`, `%carry`, and `%output` have the same type. `%cond` is `i1`.

For a loop-level condition stream `[true, true, false]`, a carry used
for an `scf.for` iter_arg observes three condition tokens and must also
receive three feedback tokens. The output stream has three values:

| Event | `%cond` consumed | `%carry` consumed | `%output` |
|-------|------------------|-------------------|-----------|
| init | none | none | `init` |
| first feedback | true | `next0` | `next0` |
| second feedback | true | `next1` | `next1` |
| reset | false | `next1` | none |

The final false-cycle feedback is a reset token. It is not a body
execution, but it is required so the op can consume the false condition
and return to `init` state.

## 5. `dataflow.invariant`

`dataflow.invariant` latches one initial value and replays it while a
condition stream remains true:

```
%output = dataflow.invariant %cond, %init : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, records the value,
  forwards it to `%output`, and transitions to `carry` state.
* In `carry` state, it waits for one `%cond : i1` token. It does not
  consume another `%init` token in this state.
* If `%cond` is true, it re-emits the recorded value and stays in
  `carry` state.
* If `%cond` is false, it emits no output, clears the recorded value,
  and returns to `init` state.

`%init` and `%output` have the same type. `%cond` is `i1`.

For a loop-level condition stream `[true, true, false]`, the output is
three copies of the initial value: one immediate copy, one for the
first true condition, and one for the second true condition. The false
condition resets the op and emits nothing.

For a zero-trip loop-level condition stream `[false]`, the output is
one copy of the initial value. That copy belongs to loop phase, not
body phase; a selector such as `dataflow.demux` can route it to the
loop-exit path, leaving the body path empty.

`dataflow.invariant` is appropriate for scalar values, index-like
values, vector values, and `none` control tokens. It is not appropriate
for frontend `memref<...>` bindings.

## 6. `dataflow.gate`

`dataflow.gate` converts a `(cond, value)` stream into a region-local
phase:

```
%after_cond, %after_value = dataflow.gate %before_cond, %before_value : T
```

The operation starts in `init` state. It always consumes
`%before_cond` and `%before_value` together when it fires.

* In `init` state, `(false, X)` emits nothing and stays in `init`.
* In `init` state, `(true, X)` emits `X` on `%after_value`, emits no
  `%after_cond` token, and transitions to `continue`.
* In `continue` state, `(true, X)` emits `true` on `%after_cond`,
  emits `X` on `%after_value`, and stays in `continue`.
* In `continue` state, `(false, X)` emits `false` on `%after_cond`,
  emits no `%after_value` token, and returns to `init`.

`%before_value` and `%after_value` have the same type. The condition
operands and results are `i1`.

For a loop-level input with `N` true tokens and one trailing false
sentinel, `%after_value` has exactly `N` tokens. `%after_cond` also has
`N` tokens when `N > 0`, but it is phase-shifted:

| Input `%before_cond` | `%after_value` | `%after_cond` |
|----------------------|----------------|---------------|
| `[false]` | `[]` | `[]` |
| `[true, false]` | `[v0]` | `[false]` |
| `[true, true, false]` | `[v0, v1]` | `[true, false]` |

The `%after_cond` result is not a validity bit for `%after_value`.
Instead, it is the local close stream for the region whose values have
already been gated:

* `%after_cond = true` means this region execution is not the last one.
* `%after_cond = false` means this region execution is the last one.

The false-cycle input value is intentionally dropped by `gate`. If a
lowering needs that value, as with the false `scf.condition` operands
that become `scf.while` loop results, it must preserve the value with a
separate projection such as `dataflow.demux`.

## 7. Compiler Usage Rules

Lowering passes that use these ops must preserve the following rules:

* Loop-level `rwc` streams keep the trailing false sentinel.
* Body-phase values are produced by `dataflow.gate` or by a construction
  with equivalent token lengths.
* Loop-carried state that needs the sentinel, including iter_args and
  hidden memory-order `none` carries, is driven by the loop-level
  `rwc`, not by a body-phase value stream.
* Body-local state whose values have already been gated may be driven by
  the body-local close stream from `dataflow.gate`.
* False-cycle values are not recovered from `dataflow.gate`; they must
  be routed separately if they are semantically needed.
* Frontend `memref<...>` values are bindings, not stream values shaped
  by these ops. Dynamic control shapes the address, data, operation,
  and `none` order streams instead.

## 8. References

* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering templates
  that use these streaming semantics.
* `docs/spec-dataflow-part-2-control.md` -- firing semantics for
  `dataflow.mux`, `dataflow.demux`, `dataflow.sync`, and
  `dataflow.constant`.
* `include/Dataflow/IR/DataflowOps.td` -- canonical operation
  definitions.
* `lib/Dataflow/IR/DataflowOps.cpp` -- verifier implementation.
* `test/dataflow/unit/stream/`, `test/dataflow/unit/carry/`,
  `test/dataflow/unit/invariant/`, `test/dataflow/unit/gate/` --
  unit-level syntax and verifier tests.

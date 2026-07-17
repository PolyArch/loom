# Loom Dataflow Part 1: Streaming Ops

This document specifies the precise token timing semantics of the
stream-shaping ops in the `dataflow` dialect:
`dataflow.stream`, `dataflow.carry`, `dataflow.invariant`, and
`dataflow.gate`.

Vector stream grouping and scalar/packed stream adaptation are specified
separately in `docs/spec-dataflow-vectorization.md`. This document owns
the scalar stream-shaping primitives that those vectorization ops build
on.

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

## 2. Phase Streams

A `phase : i1` stream is loop control, not a validity bit. For one
normally closed activation:

```
K = number of true decisions and body executions
M = K + 1
phase = true^K, false
```

The final false token is an ordinary close transition. It resets the
stateful actors driven by that phase stream, but it does not imply an IV,
body value, feedback value, or body execution. Repeated activations are
represented by concatenated phase segments; each false token closes one
segment and returns the actors to their initial states.

## 3. `dataflow.stream`

`dataflow.stream` produces a valid IV stream and a loop-level phase stream
from scalar integer recurrence operands:

```
%iv, %phase = dataflow.stream %init, %limit, %step
  {step_op = "+=", cont_cond = "<"} : iN
```

The operation starts in Idle. Idle consumes one `%init`, `%limit`, `%step`
triple and establishes a current value initialized to `%init`.

* If `cont_cond(current, limit)` is true, it emits `current` on `%iv`, emits
  `true` on `%phase`, advances the current value with `step_op`, and remains
  active.
* If `cont_cond(current, limit)` is false, it emits only `false` on `%phase`
  and returns to Idle.

The current dialect definition requires `%init`, `%limit`, `%step`, and
`%iv` to share a signless integer-like type. `%phase` is always `i1`.

For `init = 0`, `limit = 5`, `step = 1`, `step_op = "+="`, and
`cont_cond = "<"`:

| Result | Tokens |
|--------|--------|
| `%iv` | `[0, 1, 2, 3, 4]` |
| `%phase` | `[T, T, T, T, T, F]` |

For a zero-trip activation with `init = limit = 5`:

| Result | Tokens |
|--------|--------|
| `%iv` | `[]` |
| `%phase` | `[F]` |

Because `%iv` already has exactly K tokens, body address computation and
memory effects consume it directly. Pairing `%phase` and `%iv` through a
`dataflow.gate` is incorrect: the two streams have different cardinalities.

## 4. `dataflow.carry`

`dataflow.carry` is a two-state token element for loop-carried values or
hidden loop-carried `none` state:

```
%output = dataflow.carry %phase, %init, %next : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, forwards it to
  `%output`, and transitions to `carry` state.
* In `carry` state, it first inspects the head `%phase : i1` token.
* If `%phase` is true, it requires and consumes one `%next` token, consumes
  the phase token, forwards `%next` to `%output`, and stays in `carry` state.
* If `%phase` is false, it consumes only the phase token, emits no output,
  and returns to `init` state.

`%init`, `%next`, and `%output` have the same type. `%phase` is `i1`.

For `%phase = [T, T, F]`, carry consumes two next values and produces three
outputs:

| Event | `%phase` consumed | `%next` consumed | `%output` |
|-------|-------------------|------------------|-----------|
| init | none | none | `init` |
| first transition | true | `next0` | `next0` |
| second transition | true | `next1` | `next1` |
| close | false | none | none |

For one closed activation, carry consumes one init, K next values, and M
phase tokens, and emits M outputs: `[init, next0, ..., next(K-1)]`.

## 5. `dataflow.invariant`

`dataflow.invariant` latches one initial value and replays it while a
condition stream remains true:

```
%output = dataflow.invariant %cond, %init : T
```

The operation starts in `init` state.

* In `init` state, it waits for one `%init` token, records the value,
  forwards it to `%output`, and transitions to running state.
* In running state, it waits for one `%cond : i1` token. It does not
  consume another `%init` token in this state.
* If `%cond` is true, it re-emits the recorded value and stays in
  running state.
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

For a parent input with K true tokens and one trailing false close,
`%after_value` has exactly K tokens. `%after_cond` is empty when K is zero;
otherwise it has K tokens and is phase-shifted:

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

The parent false-lane input value is intentionally dropped by `gate`. If a
lowering needs that value, as with the false `scf.condition` operands
that become `scf.while` loop results, it must preserve the value with a
separate projection such as `dataflow.demux`.

## 7. Compiler Usage Rules

Lowering passes that use these ops must preserve the following rules:

* Loop-level phase streams keep the final false close token.
* `dataflow.stream` IVs already have body cardinality and are consumed
  directly; they are not paired with phase through `dataflow.gate`.
* Parent-domain carry and invariant outputs are projected through
  `dataflow.gate` before body arithmetic or memory use.
* Loop results and exit frontiers are projected from the false lane of
  `dataflow.demux %phase, %parent_value`.
* Carry feedback contains exactly K real next values. The false close
  transition never consumes a dummy feedback token.
* Body-local state whose values have already been gated may be driven by
  the body-local close stream from `dataflow.gate`.
* False-lane values are not recovered from `dataflow.gate`; they must
  be routed separately if they are semantically needed.
* Phase fanout is independent per use. One blocked consumer does not prevent
  another phase consumer from firing when its own operands are ready.
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

# Loom Dataflow Part 2: Control Ops

This document specifies the precise token timing semantics of the
control-routing ops in the `dataflow` dialect:
`dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
`dataflow.demux`.

The canonical IR source is `include/Dataflow/IR/DataflowOps.td`; the
verifier implementation lives in `lib/Dataflow/IR/DataflowOps.cpp`.
This document is the design-level companion for compiler specs that use
these ops, especially `docs/spec-compiler-part-3-dfg.md`.

## 1. Control Token Model

The control ops in this document control which token streams fire and
when they fire. They do not create implicit program order beyond their
explicit SSA token dependencies.

Multiple SSA uses of one stream are token broadcast: each use observes
the same ordered token sequence. `dataflow.mux` and `dataflow.demux`
are selective routers, not all-lane rendezvous ops. `dataflow.sync` is
the all-input rendezvous op.

An op result with no SSA uses is a dead output. It does not create a
required runtime channel, queue, or backpressure source. Implementations
may omit it, connect it to a hardware discard/disconnect path, or drop
the produced token at the producing boundary. This rule is deliberately
not represented by a separate `dataflow.drop` or `dataflow.sink` op.
It is needed for selected-control lowering, where a demux may project a
live-in stream to lanes that no selected region actually consumes.
Fabric lowerings can realize this with the PE-level `discard` /
`disconnect` controls rather than with a software-visible dataflow op.

For selectors:

* With exactly two lanes, the selector type is `i1`; `false` selects
  lane 0 and `true` selects lane 1.
* With more than two lanes, the selector type is `index`; selector
  value `k` selects lane `k`.
* A dynamic `index` selector outside the available lane range is
  invalid program behavior. Lowering must generate in-range selectors
  or prove that selectors are in range.

## 2. `dataflow.constant`

`dataflow.constant` emits a recorded typed attribute value once per
incoming control token:

```
%value = dataflow.constant %ctrl {const_value = 42 : i32} : i32
```

The op fires when one `%ctrl : none` token is available. It consumes
that token and emits one `%value` token carrying `const_value`.

There is no implicit initial constant token. A constant value appears
only when the explicit control input fires. This makes constants
phase-aware: a loop body constant must be triggered by the loop body's
per-execution control stream, while a graph-entry constant may be
triggered by the graph `ctrl_in`.

The `const_value` attribute must be typed, and its type must equal the
result type.

## 3. `dataflow.sync`

`dataflow.sync` is a variadic rendezvous:

```
%a2, %b2 = dataflow.sync %a, %b : (T, U) -> (T, U)
```

The op fires only when every input has at least one token available. It
then consumes one token from every input and emits one token on every
output simultaneously. Output `i` carries the value consumed from input
`i`.

Operand count equals result count, and operand/result types match
positionally. A zero-input, zero-output sync is syntactically legal and
has no observable token result; compiler lowering must not use it as a
control source.

`dataflow.sync` must not be used to join mutually exclusive paths. If
one path is dynamically unselected, that path produces no token, so a
sync over all paths would wait forever. Mutually exclusive tails are
joined with `dataflow.mux` using the same selector that chose the path.

## 4. `dataflow.mux`

`dataflow.mux` is an N-to-1 selective router:

```
%out = dataflow.mux %sel, %lane0, %lane1 : (i1, T, T) -> T
```

For each firing, `%sel` chooses one input lane `k`. The op fires when
both `%sel` and the selected input lane `%inputs[k]` have tokens
available. It consumes one selector token and one token from the
selected input lane, then emits that selected token on `%out`.

Non-selected input lanes are not consumed and do not need to have a
token available. Their tokens remain buffered for later selector
tokens. This selective firing rule is required for structured-control
lowering: a true branch may join through a mux even when the false lane
has no token, and loop feedback may use the body-yield lane on true
iterations while the exit/reset lane is empty.

`dataflow.mux` is not a rendezvous. If all lanes must be present before
continuation, use `dataflow.sync` before or after the mux as an
explicit dependency.

The op requires at least two input lanes. All input lanes and the output
share one type. With two lanes, `%sel` is `i1`; with more than two
lanes, `%sel` is `index`.

## 5. `dataflow.demux`

`dataflow.demux` is a 1-to-N selective router:

```
%lane0, %lane1 = dataflow.demux %sel, %in : (i1, T) -> (T, T)
```

For each firing, `%sel` chooses one output lane `k`. The op fires when
both `%sel` and `%in` have tokens available. It consumes one selector
token and one input token, then emits that input token only on
`%outputs[k]`.

Non-selected output lanes receive no token for that firing. This is the
reason `dataflow.demux` can model mutually exclusive paths: the
unselected path is silent, and any later join must use a selector-matched
`dataflow.mux`, not `dataflow.sync`.

If the selected output lane has no SSA uses, the produced token is a
dead output and follows the dead-output rule above. It must not apply
backpressure to the demux.

The op requires at least two output lanes. The input and every output
share one type. With two lanes, `%sel` is `i1`; with more than two
lanes, `%sel` is `index`.

## 6. Structured-Control Usage Rules

Lowering passes that use these ops must preserve the following rules:

* Mutually exclusive path entry uses `dataflow.demux`.
* Mutually exclusive path exit uses `dataflow.mux` with the same
  selector, or with a selector in the same dynamic phase.
* `dataflow.sync` is used only when all inputs are expected to fire on
  the same dynamic path.
* Unused projected lanes are dead outputs. They are discarded by the
  target lowering rather than represented by a separate dataflow op.
* Loop feedback muxes rely on selective firing: the true feedback lane
  and the false reset/exit lane fire on different dynamic events.
* `i1` lane order is always false lane 0, true lane 1.
* `index` lane order is always positional lane `k`.

## 7. References

* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering templates
  that use these control-routing semantics.
* `docs/spec-dataflow-part-1-streaming.md` -- timing semantics for
  streaming state ops that are often combined with these control ops.
* `include/Dataflow/IR/DataflowOps.td` -- canonical operation
  definitions.
* `docs/spec-fabric-pe.md` -- PE-level `discard` / `disconnect`
  controls used by hardware lowering for dead ports.
* `lib/Dataflow/IR/DataflowOps.cpp` -- verifier implementation.
* `test/dataflow/unit/constant/`, `test/dataflow/unit/sync/`,
  `test/dataflow/unit/mux/`, `test/dataflow/unit/demux/` --
  unit-level syntax and verifier tests.

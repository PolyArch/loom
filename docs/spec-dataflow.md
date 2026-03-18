# FCC Dataflow Dialect Specification

## Overview

FCC uses a small dataflow dialect to represent loop-derived control and
stateful stream behavior in DFG form.

Current core operations are:

- `dataflow.stream`
- `dataflow.gate`
- `dataflow.carry`
- `dataflow.invariant`

These operations are software-graph operations. They may later map onto
`fabric.function_unit` bodies inside `spatial_pe` or `temporal_pe`.

## `dataflow.stream`

`dataflow.stream` generates:

- an index stream
- a `will_continue` boolean stream

It models loop progress and iteration viability.

FCC mapping treats `dataflow.stream` as a first-class operation. It is not
equivalent to any other dataflow op.

## `dataflow.gate`

`dataflow.gate` adapts a before-region stream into a body-region stream.

It consumes:

- a value stream
- a before-condition stream

It produces:

- an after-value stream
- an after-condition stream

`dataflow.gate` is not equivalent to `dataflow.invariant`.

## `dataflow.carry`

`dataflow.carry` models loop-carried dependence.

Its behavior is stateful:

- it first emits the initial value
- it then alternates between consuming loop condition and loop-carried value
- it resets when the condition stream ends the loop

FCC mapping treats `dataflow.carry` as a first-class operation.

## `dataflow.invariant`

`dataflow.invariant` stores a loop-invariant value and replays it while the
loop condition remains true.

This operation is semantically distinct from `dataflow.gate`.

Normative mapper rule:

- `dataflow.invariant` must only match hardware that explicitly implements
  `dataflow.invariant`
- it must not be matched through an implicit compatibility alias to
  `dataflow.gate`

## Mapper Contract

For FCC mapping:

- each dataflow op above is a distinct compatibility class
- op matching is by exact operation identity unless another equivalence is
  explicitly specified in FCC spec
- no equivalence currently exists between `stream`, `gate`, `carry`, and
  `invariant`

## Visualization Contract

Focused mapper-aware visualization tests must exist for:

- `dataflow.invariant`
- a representative `stream -> gate -> carry` chain

These tests should verify:

- node mapping correctness
- routed software-edge coverage
- exported visualization payload correctness


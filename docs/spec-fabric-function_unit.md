# FCC Function Unit Specification

## Overview

`fabric.function_unit` is FCC's placeable computation body abstraction.
It is the direct bridge between software operations and hardware computation
resources inside a PE.

## Role

A function unit provides:

- native-type input and output ports
- one or more internal operations
- an optional configurable DAG shape
- a stable identity for mapping and visualization

`function_unit` is not a top-level routing resource. It lives inside
`spatial_pe` or `temporal_pe`.

Definition and instantiation rules:

- a `fabric.function_unit` definition may live at the top level, in one
  `fabric.module`, or locally inside one `spatial_pe` or `temporal_pe`
- actual instantiation is only legal inside `spatial_pe` or `temporal_pe`
- PE-local instantiation may be expressed either by:
  - a direct local `fabric.function_unit` body
  - `fabric.instance` targeting a visible `fabric.function_unit`
- a `fabric.instance` of `fabric.function_unit` inside one PE does not carry
  Fabric-edge operands or results; it only selects one local FU definition

Textual syntax follows the Fabric-wide split:

- fixed FU structure parameters such as `latency` and `interval` live in `[]`
- `fabric.mux` runtime control fields such as `sel`, `discard`, and
  `disconnect` live in braces

## Port Semantics

- FU ports use native semantic types such as `i32`, `f32`, `index`, or `none`.
- FU ports are reached through the enclosing PE's mux or demux fabric.
- FU port numbering is local to the FU and independent of PE exterior port
  numbering.

## Internal DAG

FCC allows a `function_unit` body to contain multiple operations connected as a
hardware DAG.

This DAG may be:

- fixed, for single-op or fixed multi-op FUs
- configurable, when `fabric.mux` is present

## `fabric.mux`

`fabric.mux` is the primitive that makes FU shape configurable.

Its control structure is:

- `sel`
- `discard`
- `disconnect`

The encoded field order is low-to-high:

- `sel`
- `discard`
- `disconnect`

Key rules:

- `sel` participates in tech-mapping search
- `disconnect` and `discard` are runtime configuration bits with handshake
  semantics
- the selected FU configuration is fixed before place-and-route

## `fabric.mux` Operational Semantics

`fabric.mux` is a handshake-aware routing primitive.

It supports two structural modes:

- `M:1`: multiple inputs, one output
- `1:M`: one input, multiple outputs

### `disconnect = 1`

When `disconnect = 1`, the static mux is fully inert:

- no input is accepted
- input-side `ready` behaves as permanently low
- `sel` is ignored
- this rule is identical for `M:1` and `1:M`

### `M:1` with `disconnect = 0`

The selected input is `sel`.

If `discard = 0`:

- the selected input is forwarded to the output
- non-selected inputs are not accepted
- non-selected input-side `ready` behaves as low

If `discard = 1`:

- the selected input is forwarded to the output
- non-selected inputs are accepted and drained locally
- non-selected input-side `ready` behaves as high

### `1:M` with `disconnect = 0`

The selected output is `sel`.

If `discard = 0`:

- the input is forwarded only to the selected output
- non-selected outputs remain invalid

If `discard = 1`:

- the input is accepted and drained locally
- all outputs remain invalid
- input-side `ready` behaves as high

### `1:1` Degenerate Case

A `1:1` static mux is semantically dead routing structure.

Normative rules:

- `discard` and `disconnect` must both be `0`
- the mux is treated as transparent pass-through
- mapper normalization and effective-graph extraction should behave as if the
  input and output were directly connected

## Runtime-Config Ownership

`fabric.mux` may appear in an ADG with textual runtime-config fields such
as `sel`, `disconnect`, and `discard`.

For mapper input, these fields are not authoritative final state. They are
treated as mapper-owned runtime-config hints:

- they may provide an initial default for visualization or hand-authored ADGs
- the mapper may overwrite them during tech-mapping
- the mapping result is the authoritative source of the selected FU
  configuration

This is a normative distinction between:

- structural ADG syntax
- mapper-selected runtime configuration

## Effective Graph

For mapping purposes, the FU is not represented only by its syntactic body.
Instead, the mapper reasons about the effective graph after applying the chosen
`mux` selections.

This distinction is normative. A DFG match is valid only against the effective
graph, not merely against all operations textually present in the FU body.

The effective graph may cover:

- a single software operation
- a software subgraph with multiple operations

The matching rule is structural and generic. It is not defined per named FU
such as `mac`; any FU with internal `mux` nodes participates in the same
configuration-enumeration and effective-graph extraction model.

## FU-Internal Runtime Configuration

FCC treats one `fabric.function_unit`'s runtime configuration payload as the
concatenation of all configurable body fields in body occurrence order.

Current configurable body fields are:

- `fabric.mux`
- `handshake.constant`
- `handshake.join`
- `arith.cmpi`
- `arith.cmpf`
- `dataflow.stream`

No other body operation currently contributes runtime configuration bits.

The serialized payload is low-to-high in body occurrence order. Each field uses
its own local encoding:

- `fabric.mux`: `[sel | discard | disconnect]`
- `handshake.constant`: literal result value bits, zero-extended or truncated
  to the result bit width
- `handshake.join`: one activity bit per hardware join input, ordered by FU
  body operand order; bit `1` means that input participates in the current
  software join, bit `0` means that input is ignored for this mapping
- `arith.cmpi`: 4-bit predicate encoding using the MLIR predicate enum value
- `arith.cmpf`: 4-bit predicate encoding using the MLIR predicate enum value
- `dataflow.stream`: 5-bit `cont_cond` one-hot encoding in the order
  `<, <=, >, >=, !=`

### Configurable `handshake.join`

FCC treats a `handshake.join` inside one `fabric.function_unit` as a fixed
maximum hardware join fan-in.

Normative rules:

- the hardware join fan-in equals the number of textual operands in the FU body
- the current FCC hardware/config encoding supports hardware join fan-in in the
  range `1..64`
- the emitted runtime-config field width equals that hardware fan-in
- a software `handshake.join` with smaller fan-in may map onto that hardware
  join if the mapper selects a subset of hardware inputs
- the selected subset is encoded as the `join_mask` bitfield described above

Example:

- hardware FU body contains a 4-input `handshake.join`
- software DFG contains a 3-input `handshake.join`
- mapper may bind it with `join_mask = 0b1101`

This means hardware inputs `0`, `2`, and `3` participate in the join, while
hardware input `1` is disabled for that mapped software node.

FCC does not carry Loom's earlier `output_tag` PE-body config field into
`fabric.function_unit`. Runtime tag handling is modeled explicitly through
tagged ports and tag-boundary operations instead.

## Relationship to PE Configuration

- In `spatial_pe`, only one FU is active at a time, so the PE reuses one FU
  configuration storage region across alternatives.
- In `temporal_pe`, multiple FU instances coexist, so per-FU configuration
  state persists independently of per-instruction selects.

## Relationship to Visualization

Visualization must preserve the distinction between:

- PE exterior ports
- FU border ports
- internal FU DAG

When mapping is enabled, the displayed route from a PE exterior port to a FU
port is part of the PE mux or demux fabric, not part of the FU internal DAG.

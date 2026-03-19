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

The detailed operation-class and timing model for `latency`, `interval`, and
temporal-PE completion behavior lives in
[spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md).

The detailed packing and serialization rules for FU-internal runtime
configuration live in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

Normative structural restrictions:

- all inputs must be consumed:
  every `function_unit` block argument must be consumed by at least one body
  operation
- all outputs must be driven:
  the operand count of the terminator `fabric.yield` must equal the result
  count declared by the `function_unit` type
- input-to-output passthrough is illegal:
  a `fabric.yield` operand must not be a direct block argument of the same
  `function_unit`

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
- FU-internal runtime-configurable fields live in braces

This document intentionally does not restate the full timing-class rules.
`spec-fabric-function_unit-ops.md` is the single normative source for:

- what `latency` and `interval` mean
- when `-1` is required or forbidden
- which FU bodies are treated as dedicated dataflow state machines
- how temporal-PE output draining changes observable completion time

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

## FU-Internal Configurability

FCC allows one FU body to contain configurable internal structure, primarily
through `fabric.mux` and a small set of runtime-configurable body operations.

This document intentionally does not duplicate the normative definitions for:

- which body operations are configurable
- the semantic distinction between `fabric.mux`, `handshake.mux`, and
  `arith.select`
- `fabric.mux` handshake behavior
- the low-to-high bit layout of configurable fields

These are defined in:

- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)

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

One FU contributes a runtime-configuration payload composed from the
configurable body fields that appear in its body.

The authoritative definition of:

- which fields contribute payload bits
- field-local encodings
- body-occurrence ordering
- special handling of configurable `handshake.join`

lives in [spec-fabric-config_mem.md](./spec-fabric-config_mem.md), with the
semantic meaning of each configurable body operation defined in
[spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md).

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

## Related Documents

- [spec-fabric-function_unit-ops.md](./spec-fabric-function_unit-ops.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)

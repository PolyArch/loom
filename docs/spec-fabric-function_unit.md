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
- configurable, when `fabric.static_mux` is present

## `fabric.static_mux`

`fabric.static_mux` is the primitive that makes FU shape configurable.

Its control structure is:

- `sel`
- `disconnect`
- `discard`

Key rules:

- `sel` participates in tech-mapping search
- `disconnect` and `discard` are derived from the selected effective graph
- the selected FU configuration is fixed before place-and-route

## Effective Graph

For mapping purposes, the FU is not represented only by its syntactic body.
Instead, the mapper reasons about the effective graph after applying the chosen
`static_mux` selections.

This distinction is normative. A DFG match is valid only against the effective
graph, not merely against all operations textually present in the FU body.

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

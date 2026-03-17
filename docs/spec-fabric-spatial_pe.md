# FCC Spatial PE Specification

## Overview

`fabric.spatial_pe` is a spatial compute container that holds multiple
`function_unit` instances. At runtime, one FU choice is active through the PE's
configuration state.

## Structural Model

A spatial PE contains:

- PE exterior input ports
- PE exterior output ports
- one or more `function_unit` instances
- input muxes from PE inputs to FU inputs
- output demuxes from FU outputs to PE outputs
- shared opcode and FU internal configuration state

## Port Model

PE exterior ports use structural bit types. FU ports use native semantic types.
Width adaptation occurs at the PE boundary according to the Fabric rules.

The PE-to-FU relationship is not positional:

- `PE input 0` may feed any FU input selected by configuration
- `PE output 1` may be driven by any FU output selected by configuration

Therefore PE port identity and FU port identity must both be preserved in the
mapping and visualization model.

## Configuration Layout

The spatial PE configuration contains:

- opcode selection for the active FU
- one input-mux control word per maximum FU input count
- one output-demux control word per maximum FU output count
- one FU internal config payload sized by the maximum FU config width

Each mux or demux control word contains:

- `sel`
- `disconnect`
- `discard`

## Disconnect and Discard

### Input Mux

- normal: selected PE input reaches the FU input
- disconnect: FU input is inert
- discard: selected PE input is drained without reaching the FU

### Output Demux

- normal: FU output reaches a selected PE output
- disconnect: the route is severed
- discard: FU output is drained locally

## Mapping Implications

The mapper may place at most one FU from a `spatial_pe` for a given spatial
workload context.

Even when the flattened ADG exposes FU nodes directly, the mapping result must
still be able to answer these questions for every routed software edge:

- which PE exterior input was used
- which FU input it reached
- which FU output produced the value
- which PE exterior output carried it outward

Without this information, config generation and mapping-aware visualization are
incomplete.

## Visualization Implications

When mapping is enabled:

- used FU instances may remain expanded
- unused FU instances may be collapsed to a smaller placeholder
- PE-internal routes must be rendered as PE mux or demux wiring
- PE-internal routes must not overlap FU bodies or the PE border

This is specified in more detail in [spec-viz.md](./spec-viz.md).

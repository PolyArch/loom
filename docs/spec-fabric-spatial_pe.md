# FCC Spatial PE Specification

## Overview

`fabric.spatial_pe` is a spatial compute container that holds multiple
`function_unit` instances. At runtime, exactly zero or one FU choice is active
through the PE's configuration state.

## Structural Model

A spatial PE contains:

- PE exterior input ports
- PE exterior output ports
- one or more `function_unit` instances
- input muxes from PE inputs to FU inputs
- output demuxes from FU outputs to PE outputs
- shared opcode and FU internal configuration state

Placement and definition rules:

- `fabric.spatial_pe` definitions may appear directly in the top-level module
  or in `fabric.module`
- inline `fabric.spatial_pe` instantiations may appear directly only in
  `fabric.module`
- a `spatial_pe` body may contain `fabric.function_unit` definitions and
  `fabric.instance` uses that target `fabric.function_unit`

## Port Model

PE exterior ports use structural bit types. FU ports use native semantic types.
Width adaptation occurs at the PE boundary according to the Fabric rules.

The PE-to-FU relationship is not positional:

- `PE input 0` may feed any FU input selected by configuration
- `PE output 1` may be driven by any FU output selected by configuration

Therefore PE port identity and FU port identity must both be preserved in the
mapping and visualization model.

## Configuration Layout

The spatial PE configuration is one container-local record.

Its low-to-high bit layout is:

- `spatial_pe_enable`
- `opcode`
- input mux controls
- output demux controls
- FU internal config payload

`spatial_pe_enable` is the least-significant bit. When it is `0`, the PE is
architecturally inactive and may be used for clock gating or power gating.

The remaining payload contains:

- opcode selection for the active FU
- one input-mux control word per maximum FU input count
- one output-demux control word per maximum FU output count
- one FU internal config payload sized by the maximum FU config width

Opcode numbering is derived from `function_unit` definition order inside the
`spatial_pe` body, starting from `0`.

The number of mux or demux control words is derived from:

- `max_fu_inputs` across all FUs in the PE
- `max_fu_outputs` across all FUs in the PE

Each mux or demux control field is encoded low-to-high as:

- `sel`
- `discard`
- `disconnect`

The FU-internal config payload is concatenated in two nested orders:

- `function_unit` order follows the `spatial_pe` body definition order
- within one `function_unit`, `fabric.mux` fields follow body occurrence
  order

For an unused `spatial_pe`, the default serialized state is:

- `spatial_pe_enable = 0`
- `opcode = 0`
- all input mux fields use `disconnect = 1`
- all output demux fields use `disconnect = 1`
- all FU-internal config bits are `0`

## Disconnect and Discard

### Input Mux

- normal: selected PE input reaches the FU input
- disconnect: FU input is inert
- discard is reserved in the same field shape and follows the same handshake
  meaning as other mux-like controls when used

### Output Demux

- normal: FU output reaches a selected PE output
- disconnect: the route is severed
- discard: FU output is drained locally

## Mapping Implications

The mapper must activate at most one physical `function_unit` inside one
`spatial_pe` for a given mapping result.

Normative implications:

- this is a hard legality rule, not a placement preference
- multiple software operations may still map into that one active
  `function_unit` through tech-mapping
- a mapping that requires two distinct physical `function_unit` instances from
  the same `spatial_pe` is illegal and must be rejected

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

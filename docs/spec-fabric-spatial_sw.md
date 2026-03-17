# FCC Spatial Switch Specification

## Overview

`fabric.spatial_sw` is FCC's configurable routing switch. It replaces Loom's
older `fabric.switch` naming and adds decomposable routing support.

## Hardware Parameters

Key hardware parameters are:

- input port count and widths
- output port count and widths
- `connectivity_table`
- `decomposable_bits`

`connectivity_table` defines which input positions may legally drive which
output positions.

## Runtime Configuration

Key runtime configuration is:

- `route_table`
- `discard_bit`

`route_table` is a mux-style selection model:

- each output chooses one input from its allowed connectivity set
- a single output may not accept multiple distinct inputs in one configuration
- multiple outputs may choose the same input when broadcast or replication is
  semantically required

## Decomposable Routing

When `decomposable_bits` is positive, the switch operates at sub-lane
granularity.

Important rules:

- every port width must be divisible by `decomposable_bits`
- `connectivity_table` and `route_table` are interpreted at sub-lane granularity
- output sub-lanes must be fully driven
- unused input sub-lanes may be implicitly drained

## Fill and Broadcast Rules

FCC allows a narrow stream to be replicated across multiple output sub-lanes
when a wider destination must be fully occupied.

This implies:

- one input may legally drive multiple outputs or sub-lanes
- readiness is aggregated across the replicated consumers

The converse is not allowed:

- one output or output sub-lane may not merge multiple inputs without an
  explicit temporal or arbitration mechanism

## Discard Model

`discard_bit` is per logical input port. When asserted:

- the input is drained
- the switch reports ready upstream
- the discarded traffic does not reach any output

This complements the PE-local discard model instead of replacing it.

## Visualization Implications

Mapping-aware visualization must be able to show all configured internal
`Input -> Output` connections of a spatial switch. A route highlight for one
software edge should highlight the specific internal switch route that belongs
to that edge, while the base view may still display all configured routes.

# FCC Spatial Switch Specification

## Overview

`fabric.spatial_sw` is FCC's configurable routing switch. It replaces Loom's
older `fabric.switch` naming and adds decomposable routing support.

Unlike `fabric.temporal_sw`, a `fabric.spatial_sw` never makes routing
decisions based on tag value. If its ports are tagged, the tag travels as part
of the payload only.

Placement rules:

- a `fabric.spatial_sw` definition may appear directly in the top-level module
  or in `fabric.module`
- an inline `fabric.spatial_sw` instantiation may appear directly only in
  `fabric.module`
- `fabric.instance` targeting one `fabric.spatial_sw` definition may appear
  directly only in `fabric.module`

## Hardware Parameters

Key hardware parameters are:

- input port count and widths
- output port count and widths
- `connectivity_table`
- `decomposable_bits`

Port rules:

- input port count must be in `1..32`
- output port count must be in `1..32`
- all ports must share one tag-kind
- a `spatial_sw` may be entirely non-tagged or entirely tagged
- mixing tagged and non-tagged ports is not allowed
- tagged `spatial_sw` is legal, but tagged `spatial_sw` cannot be decomposable

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
- only non-tagged `spatial_sw` may use decomposition
- `connectivity_table` and `route_table` are interpreted at sub-lane granularity
- output sub-lanes must be fully driven
- unused input sub-lanes may be implicitly drained

## Tagged Spatial Routing

For tagged ports:

- the route choice still depends only on the configured `route_table`
- the switch does not intentionally rewrite tagged shape
- the switch behaves like a tag-agnostic carrier, not like a slot-indexed
  `temporal_sw`

If a tagged path reaches the switch through narrower or wider tagged hardware
connections elsewhere in the ADG, FCC still requires the concrete runtime tag
value to be representable on every tagged port it traverses.

In particular:

- the switch does not authorize implicit runtime tag truncation
- the switch does not authorize implicit runtime tag rewriting by zero-extended
  widening
- only explicit tag-boundary operations such as `fabric.add_tag`,
  `fabric.map_tag`, and `fabric.del_tag` may change the runtime tag meaning
  seen by later hardware stages

One important FCC use case is memory or extmemory ingress:

- multiple tagged request streams may be merged through a tagged
  `fabric.spatial_sw`
- this is legal because ingress merging is tag-agnostic
- response-side tagged separation is different and still requires
  `fabric.temporal_sw`

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

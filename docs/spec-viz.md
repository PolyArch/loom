# FCC Visualization Specification

## Overview

FCC visualization presents the DFG and ADG together and, when mapping data is
available, explains how software edges traverse hardware resources.

The viewer is a self-contained HTML artifact.

Builder-produced ADGs are expected to carry explicit layout metadata bound from
the ADG's `fabric.module`.

## Rendering Stack

FCC currently uses:

- Graphviz WASM for hierarchical DFG rendering
- D3.js for ADG layout, routing overlays, and interaction

## ADG Layout Sources

ADG visualization supports two layout sources:

- inferred layout computed by the renderer
- explicit layout loaded from a sidecar JSON referenced by
  `fabric.module attributes {viz_file = "..."}` 

Normative rules:

- ADGs emitted by the FCC ADG Builder always emit a sidecar layout file and
  bind it through `viz_file`
- topology helpers such as ring, mesh, torus, chess, lattice, cube, and
  star-like generators must precompute coordinates instead of delegating
  placement to the browser
- generic Builder-produced ADGs that do not use a known topology helper must
  still emit a non-overlapping sidecar placement for all components
- for those generic Builder-produced ADGs, the non-overlapping sidecar
  placement should be computed offline during artifact generation, using a
  graph-layout pass comparable to Graphviz `neato`, rather than computed in the
  browser at HTML open time
- when explicit layout metadata is present, the renderer must prefer it over
  heuristic placement for the referenced components

The current sidecar schema is:

- top-level `version`
- top-level `components`
- each component entry contains:
  - `name`
  - `kind`
  - `center_x`
  - `center_y`
  - optional `width`
  - optional `height`
  - optional `grid_row`
  - optional `grid_col`

- top-level `routes`
- each route entry contains:
  - `from`
  - `from_port`
  - `to`
  - `to_port`
  - precomputed orthogonal `points`

The sidecar path is resolved relative to the source `fabric.mlir` path when the
`viz_file` attribute stores a relative path.

## Display Modes

The viewer should support:

- side-by-side DFG and ADG inspection
- mapping on or off
- an ADG-focused overlay mode when mapping is enabled
- simulation playback on or off when versioned trace data is embedded

Default initial view:

- when mapping data is present, the page should open as `Mapping On` with
  `Overlay`
- when mapping data is absent, the page should open as `Mapping Off` with
  `Side-by-side`

When embedded simulation trace data is present, the toolbar must also expose:

- `Simulation: On/Off`
- `Reset`
- `Stop`
- `Step`
- `Back`
- `Auto Play`

When simulation playback is active, the status display should distinguish the
hardware cycle number from the playback-frame number. The preferred wording is:

- `Cycle: <cycle> | Event Frame: <frame>/<total>`

## Base ADG Routing Requirements

Module-level hardware edges must satisfy these visual constraints:

- they do not overlap hardware bodies or the module border
- they do not share routed segments
- they may cross, but crossings should use a hop-over effect
- they should avoid gratuitous local loops, U-turns, and spurs

For topology-aware ADGs that provide explicit component coordinates, these
constraints apply on top of the supplied geometry. The renderer should not
discard explicit placement and then recompute a different base component
layout.

When Builder sidecar routes are available, the browser should consume those
precomputed routes directly rather than re-running global edge routing or
layout search at page-open time.

When mapping is off, the base ADG rendering must still draw the routed
module-level hardware edges. The renderer must not switch to a reduced
"fast-path" picture that hides hardware routing or substantially coarsens node
appearance merely because mapping overlays are disabled.

## Mapping-On Spatial PE Requirements

When mapping is enabled and a `spatial_pe` is shown:

- the PE must display routes from PE exterior input ports to the used FU input
  ports
- the PE must display routes from used FU output ports to PE exterior output
  ports
- PE-internal routes should follow the same orthogonal-routing style as
  module-level hardware edges
- unused FUs may be collapsed to a visibly smaller placeholder, not merely
  dimmed
- if a `spatial_pe` or `temporal_pe` has no mapped `function_unit` at all,
  then under `Mapping On` all of its internal `function_unit`s must be rendered
  in the collapsed state
- collapsed FUs may omit detailed ports and internal DAG rendering
- PE-internal FU layout should remain roughly square overall rather than
  degenerating into one long horizontal strip when many FUs are collapsed

## Mapping-On Spatial Switch Requirements

When mapping is enabled and a `spatial_sw` is shown:

- the base view should draw all configured internal `Input -> Output` routes
- selecting one software edge should highlight the specific switch-internal
  route used by that edge
- switch-internal rendering must reflect route-table semantics, not a guessed
  or default fanout picture

## Full-Path Highlight Contract

Selecting a software edge should highlight the whole corresponding hardware
route, including:

- module-level routed edges
- switch-internal `Input -> Output` segments
- PE-internal ingress and egress wiring

When routing passes through boundary-tagging resources, the highlighted path
must also include:

- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

For tagged memory-interface bridges:

- ingress routes may pass through `fabric.add_tag`, `fabric.map_tag`, one or
  more tagged `fabric.spatial_sw`, and optionally tagged `fabric.temporal_sw`
- egress response routes should highlight the actual tagged path, which may
  include `fabric.map_tag` before the `fabric.temporal_sw` split stage and
  `fabric.del_tag` only at the final non-tagged boundary
- the renderer must not assume one fixed bridge micro-topology for memory or
  extmemory traffic

For Builder-produced ADGs that explicitly instantiate memory bridges:

- bridge components such as `fabric.add_tag`, `fabric.map_tag`,
  `fabric.del_tag`, bridge-local `fabric.spatial_sw`, and bridge-local
  `fabric.temporal_sw` should appear at their precomputed positions around the
  corresponding `fabric.memory` or `fabric.extmemory`
- ingress-side bridge components should render on the logical input side of the
  memory-family component
- egress-side bridge components should render on the logical output side of the
  memory-family component

Memory-family component ports should render with family names instead of only
numeric indices. For tagged families, the label should append the tag width in
the form `name[iK]`, such as `ld_addr[i3]`.

The highlighted hardware path must be semantically continuous from the software
edge's source to its destination.

## Mapping-On Function Unit Requirements

When one or more software operations are tech-mapped into one configurable FU:

- the FU should remain visually identifiable as one hardware component
- software edges that become FU-internal edges should not be shown as unrouted
  inter-component failures
- the selected `fabric.mux` choice should be visible in the FU view
  whenever that choice materially changes the effective graph

## Interaction Requirements

- component hit regions should not swallow border-port clicks unnecessarily
- shrinking component hit regions relative to the visible border is allowed and
  recommended
- ports and routed edges should remain directly selectable
- releasing the center divider after a panel resize should trigger a fresh fit
  of both panes to the new viewport

## Simulation Playback Requirements

The HTML artifact may embed a sibling versioned JSON trace as `SIM_TRACE_DATA`.
When trace data is present and has a supported `version`, the renderer must:

- group events into deterministic playback frames
- support reset-to-frame-zero playback
- support forward stepping and reverse stepping
- support autoplay
- highlight firing hardware modules
- highlight active routed hardware edges and switch routes
- highlight relevant boundary ports and memory bindings
- highlight mapped DFG nodes corresponding to firing hardware modules

If the embedded trace version is unsupported, the page must disable simulation
playback instead of silently attempting to interpret the trace.

## Relationship to Mapping Output

Visualization correctness depends on the mapping report carrying enough
component-local route identity. If the mapping payload omits PE-local or
switch-local route choices, the visualization cannot be considered fully
specified.

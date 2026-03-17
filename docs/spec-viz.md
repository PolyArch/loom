# FCC Visualization Specification

## Overview

FCC visualization presents the DFG and ADG together and, when mapping data is
available, explains how software edges traverse hardware resources.

The viewer is a self-contained HTML artifact.

## Rendering Stack

FCC currently uses:

- Graphviz WASM for hierarchical DFG rendering
- D3.js for ADG layout, routing overlays, and interaction

## Display Modes

The viewer should support:

- side-by-side DFG and ADG inspection
- mapping on or off
- an ADG-focused overlay mode when mapping is enabled

## Base ADG Routing Requirements

Module-level hardware edges must satisfy these visual constraints:

- they do not overlap hardware bodies or the module border
- they do not share routed segments
- they may cross, but crossings should use a hop-over effect
- they should avoid gratuitous local loops, U-turns, and spurs

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
- collapsed FUs may omit detailed ports and internal DAG rendering

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

The highlighted hardware path must be semantically continuous from the software
edge's source to its destination.

## Mapping-On Function Unit Requirements

When one or more software operations are tech-mapped into one configurable FU:

- the FU should remain visually identifiable as one hardware component
- software edges that become FU-internal edges should not be shown as unrouted
  inter-component failures
- the selected `fabric.static_mux` choice should be visible in the FU view
  whenever that choice materially changes the effective graph

## Interaction Requirements

- component hit regions should not swallow border-port clicks unnecessarily
- shrinking component hit regions relative to the visible border is allowed and
  recommended
- ports and routed edges should remain directly selectable

## Relationship to Mapping Output

Visualization correctness depends on the mapping report carrying enough
component-local route identity. If the mapping payload omits PE-local or
switch-local route choices, the visualization cannot be considered fully
specified.

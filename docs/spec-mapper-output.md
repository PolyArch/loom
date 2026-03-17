# FCC Mapper Output Specification

## Overview

A successful FCC mapping produces machine-readable, human-readable, and
visualization-oriented outputs.

## Output Families

The main mapping outputs are:

- `<base>.map.json`
- `<base>.map.txt`
- `<base>.viz.html`

Additional config or simulation artifacts may be produced by later stages.

## JSON Mapping Report

The JSON report is the authoritative structured output for downstream tools.

### Core Sections

At minimum, the report should include:

- `seed`
- `node_mappings`
- `edge_routings`
- `port_table`
- `memory_regions`

### Required Semantics

`node_mappings` must identify:

- software node id
- mapped hardware node id
- hardware resource name
- enclosing PE identity when applicable

`edge_routings` must identify:

- software edge id
- a route description whose step semantics are reconstructable
- whether the edge is routed through inter-component hardware or absorbed as an
  intra-FU edge by tech-mapping

`port_table` must identify:

- flat port id
- component kind
- component name
- local port index
- direction

`memory_regions` must identify:

- selected hardware memory node id and name
- hardware memory kind
- hardware `num_region`
- for each occupied region:
  - software memory node id
  - backing software memref argument index
  - selected bridge or tag lane
  - load/store counts
  - `elem_size_log2`
- exported `addr_offset_table`

## Extended Visualization Payload

FCC should evolve the JSON report to also expose component-local routing facts.
Recommended sections are:

- `switch_routes`
  - per switch, list of configured `input_index -> output_index` selections
- `pe_routes`
  - per PE, list of selected ingress and egress mux or demux bindings
- `fu_configs`
  - selected effective FU configuration per mapped hardware FU
  - software nodes absorbed into that FU
  - selected `static_mux` fields such as `sel`

For memory-oriented visualization, `memory_regions` is not optional in
practice. It is the authoritative bridge between:

- software memref arguments
- software memory ops
- chosen hardware memory interfaces
- region-table configuration

These extensions are strongly recommended because they remove ambiguity from
mapping-aware visualization.

## Text Mapping Report

The text report is for human inspection. It should summarize:

- node placements
- edge routes
- PE utilization
- important diagnostics or omissions

The text report is informative, not the primary data interface.

## Visualization HTML

The visualization HTML is self-contained and embeds:

- ADG data
- DFG data
- mapping data
- renderer assets

The HTML is a consumer of mapping JSON semantics, not a separate source of
mapping truth.

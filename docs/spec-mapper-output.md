# FCC Mapper Output Specification

## Overview

A successful FCC mapping produces machine-readable, human-readable, and
visualization-oriented outputs.

## Output Families

Mapping outputs are mixed software-plus-hardware artifacts and therefore use
the naming family `<dfg>.<adg>.*`.

The main mapping outputs are:

- `<dfg>.<adg>.map.json`
- `<dfg>.<adg>.map.txt`
- `<dfg>.<adg>.viz.html`
- `<dfg>.<adg>.config.json`
- `<dfg>.<adg>.config.bin`
- `<dfg>.<adg>.config.h`

Additional config or simulation artifacts may be produced by later stages.

## Config JSON

`<dfg>.<adg>.config.json` is the authoritative structured summary of serialized
runtime configuration.

It must include:

- flattened `words`
- per-slice metadata with `word_offset`, `word_count`, and completeness
- container slices for `spatial_pe` and `temporal_pe`, not only primitive
  routing or memory nodes

For `spatial_pe`, the slice low bit is `spatial_pe_enable`, followed by opcode,
PE input mux controls, PE output demux controls, and the selected FU-internal
config payload.

For `temporal_pe`, the slice low region is instruction memory ordered by slot,
and the high region stores persistent per-function_unit internal config bits.

## JSON Mapping Report

The JSON report is the authoritative structured output for downstream tools.

### Core Sections

At minimum, the report should include:

- `seed`
- `node_mappings`
- `edge_routings`
- `port_table`
- `temporal_registers`
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
- whether the edge is an internal temporal-register dependency rather than an
  inter-component route

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

`temporal_registers` must identify:

- temporal PE name
- software edge id
- allocated register index
- writer software and hardware node ids
- reader software and hardware node ids
- writer output index and reader input index

## Extended Visualization Payload

The current FCC `map.json` report also exposes component-local routing facts
needed by visualization and config inspection.

Current sections include:

- `switch_routes`
  - per switch, list of configured `input_index -> output_index` selections
- `pe_routes`
  - per PE, list of selected ingress and egress mux or demux bindings
- `fu_configs`
  - selected effective FU configuration per mapped hardware FU
  - software nodes absorbed into that FU
  - selected `mux` fields such as `sel`
  - configurable software-op fields such as:
    - `handshake.constant` literal value
    - `handshake.join` active-input bitmask `join_mask`
    - `arith.cmpi` / `arith.cmpf` predicate
    - `dataflow.stream` continuation condition
- `tag_configs`
  - per-tag-boundary runtime configuration
  - `fabric.add_tag` exports one constant `tag`
  - `fabric.map_tag` exports `table_size`, tag widths, and the full structured
    table entries `[valid, src_tag, dst_tag]`
- `fifo_configs`
  - per-FIFO runtime configuration
  - only present for bypassable FIFOs
  - exports whether the FIFO is currently bypassed
- `temporal_registers`
  - explicit register-backed dependencies inside `temporal_pe`
  - the assigned register index per writer result

For memory-oriented visualization, `memory_regions` is not optional in
practice. It is the authoritative bridge between:

- software memref arguments
- software memory ops
- chosen hardware memory interfaces
- region-table configuration

These sections are part of the current report family because they remove
ambiguity from mapping-aware visualization and config inspection.

Current `edge_routings` entries also classify the software edge into one of:

- `routed`
- `unrouted`
- `intra_fu`
- `temporal_reg`

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

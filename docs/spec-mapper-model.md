# FCC Mapper Model Specification

## Overview

This document defines the logical model and hard constraints for FCC mapping.

## Graph Families

FCC mapping uses three related graph views:

### Software Graph

The DFG represents software operations and edges.

### Hardware Graph

The ADG represents hardware resources and structural connectivity.

### Flattened Mapper View

The mapper may flatten PE containers so FUs become directly placeable nodes.
However, flattening must not destroy the information required to reconstruct:

- PE exterior input and output identities
- FU containment in a PE
- switch port identities
- route-table ownership

This is a hard contract, not an optimization hint.

### Contracted Tech-Mapped View

After FU-configuration selection, the mapper may build a contracted DFG view in
which one matched software subgraph is represented as one placeable unit.

This contracted view is a planning artifact for Layer 3. It must preserve:

- the identity of every original software node covered by the unit
- the original software edges that become FU-internal edges
- the hardware FU instance and locked FU configuration associated with the
  contracted unit
- the external software-to-hardware port bindings of the contracted unit

Layer 3 may place and route the contracted view, but final reporting must be
expanded back to the original software DFG.

## Canonical Route Semantics

For FCC, a routed software edge is not just an arbitrary port sequence. It must
retain enough meaning to reconstruct the component-local path.

Canonical route steps may include:

- module or boundary output to switch input
- switch input to switch output
- switch output to PE exterior input
- PE exterior input to FU input
- FU output to PE exterior output
- PE exterior output to switch input
- switch output to module or boundary input

Not every edge uses all step types, but the chosen representation must preserve
which step occurred where.

## Spatial PE Contract

Even if the flattened graph exposes only FU nodes as placeable compute
resources, the mapping state must still preserve PE-local routing identity.

For every routed software edge touching a spatial PE, the mapper must be able to
answer:

- which PE input port was selected for ingress
- which FU input port consumed the value
- which FU output port produced the value
- which PE output port was selected for egress

## Spatial Switch Contract

For every configured spatial switch:

- each output or output sub-lane selects at most one input or input sub-lane
- multiple outputs may select the same input when broadcast or replication is
  legal
- route legality is checked against `connectivity_table`

Any mapping state that lets two distinct software edges claim conflicting input
choices for the same spatial-switch output is invalid.

## Memory Interface Contract

Memory-interface placement is a real mapping choice.

For each software memory node, the mapper must preserve:

- which hardware memory interface was selected
- which region slot of that hardware interface was assigned
- which tag lane or bridge lane was assigned when tagged memory routing is used
- which software memref argument backs that region
- how software operand and result order was bridged to the hardware family
  order of load-address, store-address, store-data and load-data, load-done,
  store-done

For memory compatibility:

- software and hardware memrefs are matched by element width
- software element width may be less than or equal to hardware interface
  element width
- exact integer-vs-float element kind is not a compatibility blocker

The memref boundary edge may later be exported as a direct module-interface
binding, but that export is downstream of placement. It must not be interpreted
as evidence that memory-interface selection was trivial or predetermined.

## Hard Constraints

FCC uses the following hard-constraint families:

- `C1 Node compatibility`: the DFG op must match the selected FU effective graph
- `C2 Type compatibility`: data widths and type rules must be satisfied
- `C3 Route legality`: every route step must follow legal directed connectivity
- `C4 Capacity`: exclusive resources may not be double-booked
- `C5 Temporal legality`: slot, tag, and register limits must hold
- `C6 Config encoding`: config fragments must fit the target encodings
- `C7 Decomposable fill`: every decomposable output lane must be driven
- `C8 PE exclusivity`: one spatial PE may host at most one active spatial FU
- `C9 FU config consistency`: one physical temporal FU must not require
  incompatible internal configurations simultaneously

## Runtime-Config Hint Contract

An ADG may carry textual runtime-config fields on hardware ops, especially on
`fabric.static_mux`.

For mapping semantics:

- these fields are hints, not locked decisions
- Layer 2 may overwrite them when selecting an effective FU graph
- Layer 3 must treat the Layer-2 result as fixed and must not reselect FU
  internal configuration
- a degenerate `1:1` `fabric.static_mux` is treated as transparent routing and
  should not survive as a meaningful tech-mapping decision

This contract exists so hand-authored ADGs remain legal while the mapper still
owns the final runtime configuration.

## Intra-FU Edge Semantics

When tech-mapping contracts a software subgraph into one physical FU, some
original software edges become intra-FU edges.

For such edges:

- they are valid mapped edges
- they do not require a separate inter-component hardware route
- reports and visualization must distinguish them from unrouted failures
- config generation must attribute them to the selected FU configuration, not to
  switch or PE-exterior routing

## Output-Oriented Mapping Requirements

The mapper model must be rich enough to support:

- config generation
- textual mapping reports
- mapping-aware visualization

In practice, that means the model must carry more than bare reachability. It
must also preserve component-local route choices.

This requirement is especially important for:

- `spatial_sw` internal `Input -> Output` visualization
- `spatial_pe` internal mux and demux visualization
- full-path highlight of one software edge across module, switch, and PE views

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

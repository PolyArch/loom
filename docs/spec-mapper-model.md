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

This information is also required for container-level config generation.
A legal spatial mapping must be reconstructable into:

- `spatial_pe_enable`
- active FU opcode
- PE input-mux selections
- PE output-demux selections
- selected FU-internal config bits

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

When tagged routing is involved, the mapper computes runtime tag values for the
mapped software flows and validates them against the hardware `tagWidth`
already declared by the ADG. The mapper does not infer tagged versus
non-tagged shape or hardware tag width.

The comparison point is the runtime tag value observed at the actual hardware
resource being shared, not only the source-side software notion of tag. This
matters because FCC hardware connections allow tagged width mismatch with LSB
alignment:

- tagged wide to tagged narrow truncates high tag bits
- tagged narrow to tagged wide zero-extends high tag bits

Therefore, if two software flows share one tagged hardware edge or one tagged
routing output, the mapper must compare the runtime tag values observed at that
resource after any earlier `add_tag`, `map_tag`, and width adaptation along
the routed hardware path.

For memory-family routes, this comparison uses the bridge-expanded export path,
not only the truncated boundary path used internally during placement and
routing. Shared tagged resources inside recovered bridge suffixes or prefixes
must still participate in conflict detection.

For memory-family routing, the runtime tag value is attached to the specific
software lane carried by one DFG edge, not merely to the region base lane.
If `fabric.map_tag` appears on the routed path, later tag-dependent hardware
must observe the remapped runtime tag value. `fabric.map_tag` may also change
the declared tag width at that explicit boundary, but it must not change the
value payload width.

Bridge-boundary recovery for tagged memory families is based on route meaning,
not on one fixed helper-op pattern. A legal bridge may terminate at:

- an explicit `fabric.add_tag` / `fabric.del_tag` boundary
- or a tagged route-stage boundary port when the adjacent compute-side region
  already remains tagged

When mapper propagates runtime tags for a memory-family route, an explicit tag
already carried on the routed hardware path has priority. The software lane id
acts only as a fallback source when the path has not attached any runtime tag
yet.

For `fabric.temporal_sw`, one observed runtime tag may not require multiple
different input-to-output transitions within the same switch instance.
Designs that collapse distinct software flows to one observed tag before a
temporal split must be rejected.

For memory compatibility:

- software and hardware memrefs are matched by element width
- software element width may be less than or equal to hardware interface
  element width
- exact integer-vs-float element kind is not a compatibility blocker

The memref boundary edge may later be exported as a direct module-interface
binding, but that export is downstream of placement. It must not be interpreted
as evidence that memory-interface selection was trivial or predetermined.

## Temporal PE Contract

For every configured temporal PE, the mapper must preserve enough information
to serialize:

- slot validity
- instruction tag
- opcode
- operand-routing state
- input-mux selections
- output-demux selections
- result-tag defaults
- internal temporal register assignments
- persistent per-function_unit config state

For current FCC register-backed temporal edges:

- an internal dependency between two software nodes mapped into the same
  `temporal_pe` is represented as a `temporal_reg` edge, not as a routed
  inter-component hardware path
- register assignment is keyed by the writer software output port
- one writer may feed multiple readers through the same register
- total distinct writer output ports mapped this way must not exceed
  `num_register`

## Hard Constraints

FCC uses the following hard-constraint families:

- `C1 Node compatibility`: the DFG op must match the selected FU effective graph
- `C2 Type compatibility`: data widths and type rules must be satisfied
- `C3 Route legality`: every route step must follow legal directed connectivity
- `C4 Capacity`: exclusive resources may not be double-booked
- `C5 Temporal legality`: slot, tag, and register limits must hold
- `C6 Config encoding`: config fragments must fit the target encodings
- `C7 Decomposable fill`: every decomposable output lane must be driven
- `C8 PE exclusivity`: one spatial PE may host at most one active physical
  spatial FU, and the mapper must reject placements that violate this
- `C9 FU config consistency`: one physical temporal FU must not require
  incompatible internal configurations simultaneously

For op compatibility:

- `dataflow.invariant` and `dataflow.gate` are distinct compatibility classes
- the mapper must not treat one as an alias of the other

Current implementation note:

- the mapper now emits an explicit failure when one forced temporal placement
  would require incompatible `fabric.mux` settings on the same physical
  `function_unit`
- repeated temporal reuse of one physical `function_unit` with identical
  internal config is not yet expanded into multiple slots by the current
  mapper

## Runtime-Config Hint Contract

An ADG may carry textual runtime-config fields on hardware ops, especially on
`fabric.mux`.

For mapping semantics:

- these fields are hints, not locked decisions
- Layer 2 may overwrite them when selecting an effective FU graph
- Layer 3 must treat the Layer-2 result as fixed and must not reselect FU
  internal configuration
- a degenerate `1:1` `fabric.mux` is treated as transparent routing and
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

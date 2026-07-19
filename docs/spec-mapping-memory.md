# Mapping Memory Boundary

This document specifies confirmed memory ownership across Dataflow,
TechMapping, SpatialMapping, and SystemMapping. It does not define the still-
open persistent physical memory record schema.

## Canonical Memory Order

The Canonical Dataflow Program owns logical memory ordering. Required
happens-before relations are ordinary typed event edges in the canonical
memory-order event network. Mapping must not duplicate them as `MemoryOrder`
records, infer them from textual order, or replace them with a schedule-local
ordering authority.

Physical contention may add stalls, but it cannot weaken or recreate the
software event network. The final verifier proves that selected physical
mechanisms preserve every logical obligation.

## TechMapping Memory Realization

TechMapping owns the selected Memory Realization:

* exact load/store actor coverage and logical-root association;
* selected Fabric Memory Semantic Encoding;
* actor-to-operation-template correspondence;
* exact actor and graph boundary correspondence;
* one-beat access capability obligations; and
* exact internal-edge witnesses pairing a canonical software endpoint pair
  with a selected Fabric internal connection.

An internal-edge witness uses the exact Dataflow artifact identity plus typed
producer and consumer endpoints. It does not use an edge number, symbol, path,
or list position. Actors sharing one Memory Realization do not make every edge
between them internal; only witnessed selected connections are absorbed.

## Spatial Memory Realization

SpatialMapping selects concrete `fabric.mem` occurrences, operation
attachments, route-tree branches, physical buffers, address/service choices,
configuration, tags, and event-relative resource use where those facts are not
mechanically derivable.

There is no independent request-route and response-route authority. Each
transport obligation is a branch of the same logical-net and Route Tree model
used for other software transfers, with endpoints derived from the selected
Memory Realization and Fabric service structure. PE-local or memory-local
traversal is not free; it must be represented by explicit Fabric connectivity
or absorbed by the selected internal-connection witness.

SpatialMapping must not copy the canonical memory-order network, create a
second memory binding that competes with Memory Realization or service
selection, or store local QoR estimates.

## System Memory Boundary

SystemMapping uses the confirmed `ExecutionBinding`, `ServiceRealization`, and
`ResourceUse` families. It connects SpatialCore-local service use to exact
system-visible services and transport without allowing runtime remapping. The
exact service ownership chain, address fields, coherence fields, and
SystemMapping cardinality remain open.

## Implemented Boundary

The neutral C++ verifier implements TechMapping Memory Realization legality.
`FrozenRealizationGraph` derives concrete memory occurrence domains, external
operation-port demands, deterministic logical nets, selected internal-edge
absorption, and logical-root service obligations. These are ephemeral native
projections, not persistent SpatialMapping records.

## Validation

Current anchor tests cover exact internal-edge witnesses, foreign Dataflow
references, duplicate endpoint-pair rejection, multi-sink fanout, deterministic
freeze, memory occurrence domains, access capability, and service consistency.
Persistent address, service, route, buffer, tag, and resource-use tests wait
for their closed schemas.

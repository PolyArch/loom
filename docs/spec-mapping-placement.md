# Mapping Placement Boundary

This document specifies confirmed placement ownership. It does not define the
still-open persistent SpatialMapping or SystemMapping record syntax.

## TechMapping Authority

TechMapping owns target-specific actor grouping, selected Compute and Memory
Realizations, selected semantic encodings, and exact software boundary
correspondences. Spatial placement consumes that immutable result.

Spatial placement must not rematch raw Dataflow and Fabric inputs, recreate
`dataflow.subgraph`, reinterpret operation compatibility, or choose a new
semantic encoding. The canonical `dataflow.subgraph` operation has been
deleted and is not a placement subject or fallback authority.

## Implemented Freeze Boundary

`FrozenRealizationGraph` derives ephemeral base domains from a validated
TechMapping and the validated Fabric occurrence view. It does not select a
placement.

Fabric distinguishes PE, FU, and memory occurrences. The confirmed structural
references are:

```text
FabricFuOccurrenceRef =
  (FabricPeOccurrenceRef, FuId)

InstructionContextRef =
  (FabricPeOccurrenceRef, ContextOrdinal)

FabricMemoryOccurrenceRef =
  MemoryOccurrenceId
```

`FuId` names the Fabric-owned implementation, and the PE component is the
mechanical parent of the concrete FU occurrence. A Spatial PE owns context
ordinal zero. A Temporal PE owns the ordinal range declared by its
`num_instruction` capability. Contexts do not have optional slots, sentinels,
parallel-only types, or independent entity identity.

Compute candidate domains retain FU and parent-context correlation. They do
not form independent FU and PE domains whose Cartesian product could create a
cross-PE candidate. Memory candidate domains contain concrete `fabric.mem`
occurrences matching the Memory Semantic Encoding selected by TechMapping.

## Spatial Placement Choices

The confirmed selected relations are:

```text
ComputeRealizationRef -> selected FabricFuOccurrenceRef
ComputeRealizationRef -> selected InstructionContextRef
MemoryRealizationRef  -> selected FabricMemoryOccurrenceRef
```

The final verifier must prove that each selected FU occurrence and instruction
context have the same parent PE. Selecting only a PE is insufficient because
it loses the selected FU implementation. Compute and Memory placements remain
distinct typed families; they are not hidden behind a generic realization or
resource reference.

Spatial placement does not own thread-to-AccCore or graph-launch-to-
SpatialMapping binding. Those are SystemMapping concerns. It also does not
persist local score, estimated latency, utilization, compatibility strings,
or consumer-specific fallback markers.

## System Placement Boundary

The confirmed persistent SystemMapping families are `ExecutionBinding`,
`ServiceRealization`, and `ResourceUse`. Their exact fields, cardinality, and
lineage remain open. In particular, an InstructionCore-only thread must not be
represented by a dummy SpatialMapping.

## Unresolved Schema

Exact SpatialMapping bindings, persistent Mapping MLIR syntax, resource-time,
tags, buffers, memory service records, and SystemMapping cardinality are not
defined here. Implementations must not create placeholder records or generic
wrappers to anticipate them.

## Validation

Current tests cover typed occurrence identity, FU-parent linkage, correlated
instruction-context domains, memory occurrence domains, foreign references,
and deterministic freeze ordering. Persistent placement tests must wait for
the corresponding schema and final verifier contract.

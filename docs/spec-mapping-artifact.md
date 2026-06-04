# Mapping Artifact

This document specifies the target software-to-hardware mapping
artifact used by Loom PnR, CGRA-sim, RTL-oriented lowering, hardware
estimation, and DSE. PnR is specified in `docs/spec-pnr.md` and is the
primary producer of mapping artifacts. The mapping artifact is
independent from both software dataflow IR and hardware Fabric ADG.

## Purpose

PnR takes software dataflow IR plus a selected Fabric hardware
description and produces a concrete mapping candidate. That candidate
must be represented as a standalone artifact rather than being written
back into `dataflow.graph`, `dataflow.thread`, `dataflow.subgraph`, or
`fabric.system`.

The artifact records the chosen mapping. It does not record PnR
internal search queues, rejected internal candidates, or mutable solver
state. DSE-level comparisons belong in a mapping-set manifest.

The artifact is the source of truth for:

* which software execution units run on which hardware resources;
* which software values and control edges use which hardware routes;
* which software memory operations bind to which memory resources;
* which resources are temporally shared;
* which schedule slots, temporal tags, queues, and buffers are required;
* which diagnostics explain rejected or partial mappings;
* which metrics describe quality, cost, and estimated performance.

Fabric ADG remains the source of truth for hardware architecture.
Dataflow IR remains the source of truth for software semantics. The
mapping artifact records one chosen relation between them.

## Core Rule

A mapping artifact must not mutate the software or hardware IR it
references. It may refer to both IRs by stable symbols, paths, versions,
and fingerprints. It must not introduce hardware nodes, hardware links,
software operations, or software edges that are absent from the
referenced inputs.

If a mapping requires a hardware structure that does not exist, the
PnR tool must either reject the mapping or produce a new hardware
candidate through the hardware DSE flow. It must not silently encode
that missing hardware inside the mapping artifact.

## Artifact Granularity

One mapping artifact represents one concrete mapping candidate for one
software workload shape against one hardware system.

A DSE run may produce many mapping artifacts. A mapping-set manifest may
reference those artifacts and record objective functions, search
configuration, Pareto status, selected candidate, and summary metrics.
The manifest does not replace the per-candidate artifact.

## Inputs Referenced By A Mapping

A mapping artifact header must identify:

* the software module or dataflow root being mapped;
* the selected `fabric.system`;
* every referenced `fabric.module` template when internal SpatialCore
  resources are referenced;
* tool version or schema version;
* optional fingerprints for the software IR and hardware IR;
* the workload shape or profile input used by PnR, when the mapping is
  workload-specific.

Fingerprints are optional for early implementation but required for
reproducible DSE. If present, they must be checked by consumers before
using the artifact.

## Reference Model

All references are symbolic and stable. A reference may include a symbol
name, an operation path under that symbol, a result or operand ordinal,
and an optional fingerprint. Line numbers and source-file offsets are
not stable references.

Software references may name:

* `dataflow.thread` definitions and launches;
* logical thread instance domains;
* `dataflow.graph` definitions and launches;
* `dataflow.subgraph` operations;
* dataflow operations;
* dataflow SSA values;
* producer-consumer value edges;
* control, done, and memory-order token edges;
* logical memref regions and partitioned-data regions.

Hardware references may name:

* a `fabric.system`;
* `fabric.node` symbols;
* `fabric.external_port` symbols;
* node ports and channels;
* external-port channels;
* `fabric.link` operations;
* `fabric.module` symbols;
* SpatialCore template resources such as `fabric.pe`, `fabric.fu`,
  `fabric.mem`, `fabric.switch`, and `fabric.boundary` when the mapping
  reaches inside a SpatialCore template.

## Required Record Families

The artifact is a collection of typed records. Each record family is
optional only when the mapped workload does not need that family.

### Thread Binding

Thread binding maps logical `dataflow.thread` instance domains to
physical `acc_core` nodes and execution batches. It may be explicit for
small domains or parametric for large domains.

Parametric binding must describe a deterministic relation from logical
thread coordinates to hardware nodes and optional batch indices. It
must not assume mesh coordinates unless those coordinates are present as
optional visualization metadata or explicit hardware metadata.

### Graph Binding

Graph binding maps each selected `dataflow.graph` launch or graph
instance to a SpatialCore execution context inside an `acc_core` node.
It identifies the `acc_core` node, the referenced `fabric.module`
template, and any graph-level schedule context needed by CGRA-sim.

### Operation Binding

Operation binding maps dataflow operations, subgraphs, or software
compute regions to SpatialCore resources. Depending on mapping depth,
the target may be a `fabric.fu`, `fabric.pe`, `fabric.mem`, or another
explicit resource inside a referenced `fabric.module`.

Operation binding records temporal sharing when multiple software
entities use the same exclusive hardware resource at different schedule
slots or under different temporal tags.

### Edge Route

Edge-route records map producer-consumer software edges to ordered
hardware route segments. A route segment may reference system-level
links, channel endpoints, SpatialCore switch paths, boundary resources,
or buffers.

The ordered route must be contiguous: the destination endpoint of one
segment must match the source endpoint of the next segment, accounting
for explicit adapters and boundary resources. A route must not invent
hardware connectivity absent from Fabric ADG.

### Schedule Binding

Schedule records assign cycle offsets, initiation intervals, temporal
slots, temporal tags, reconfiguration slots, or batch indices when they
are required by the selected hardware mapping.

Schedule records belong in the mapping artifact, not in `fabric.system`
and not in software dataflow IR. Hardware may declare temporal capacity
or tag width, but this artifact assigns workload-specific temporal use.

### Buffer Binding

Buffer records bind software queues, dataflow value streams, control
tokens, and memory-order tokens to physical buffers, FIFOs, memories, or
queue resources. They also record depth, initial occupancy, and
backpressure policy when the simulator or RTL path needs those facts.

### Memory Binding

Memory-binding records map dataflow loads, stores, memref regions, and
partitioned-data regions to physical address spaces, terminal memory
target ports, cache or coherence domains, and address ranges.

A memory binding must reference a memory-capable hardware port and must
respect the address-space, terminal-range, and coherence-domain rules in
`docs/spec-fabric-system-adg.md`.

### Resource Sharing

Resource-sharing records identify exclusive hardware resources used by
multiple software entities. They must name the sharing policy, the
schedule or tag partition that makes the sharing legal, and any
conflict class that CGRA-sim or RTL lowering must preserve.

### Diagnostics

Diagnostics are part of the artifact when a mapping is partial,
rejected, or produced under degraded assumptions. Diagnostics must name
the software object, the hardware object when one was involved, and the
legality or resource rule that caused the diagnostic.

### Metrics

Metrics record objective values and estimates associated with the
mapping candidate. Baseline metrics include cycle estimate, resource
utilization, route length, buffer usage, memory traffic, temporal reuse,
unmapped software count, and diagnostic count.

Metrics may include DFG-sim report references, CGRA-sim report
references, simulation comparison report references, or FPA estimates
when available. DFG-sim, CGRA-sim, and comparison contracts are
specified in `docs/spec-dfg-sim.md`, `docs/spec-cgra-sim.md`, and
`docs/spec-simulation-comparison.md`. Metrics are evidence for DSE and
reporting; they do not change mapping legality.

## Visualization Metadata

The mapping artifact may carry optional visualization metadata. This
metadata supports GUI and human inspection. It must not affect software
semantics, hardware semantics, mapping legality, simulator behavior, RTL
generation, or FPA estimation.

Visualization metadata may describe:

* grouping and labels for mapped software units;
* highlighting for placed operations and routed edges;
* overlay colors, style classes, and visibility categories;
* preferred views such as system view, thread-instance view, graph view,
  route view, memory view, and temporal schedule view;
* mapping overlays that connect software objects to hardware objects.

Visualization metadata may reference visualization layouts defined in
Fabric ADG. For example, a hardware system may define a two-dimensional
grid layout for a mesh-like accelerator array or a three-dimensional
grid layout for a stacked topology. The mapping artifact may then
overlay placements and routes onto that layout.

If no visualization metadata is present, a GUI must still be able to
render the artifact using graph-layout algorithms over the explicit
software references, hardware references, and mapping records.

## Mapping-Set Manifest

A mapping-set manifest is an optional DSE companion artifact. It
references multiple mapping artifacts and records:

* the shared software and hardware inputs;
* the search configuration and objective functions;
* the candidate list;
* rejected candidate summaries;
* selected candidate or Pareto set;
* aggregate metrics and comparison tables.

The manifest must not duplicate detailed placement, routing, schedule,
or memory-binding records. Those remain in the per-candidate mapping
artifacts.

## Validation

A mapping artifact verifier must check:

* referenced software symbols and paths resolve;
* referenced hardware symbols, ports, channels, links, and internal
  resources resolve;
* referenced fingerprints match when fingerprints are present;
* placed software entities are placed on compatible hardware resources;
* exclusive resources are not double-booked in the same schedule slot
  or temporal tag;
* edge routes are contiguous and use existing hardware connectivity;
* route endpoints match the placement of producer and consumer objects;
* memory bindings use memory-capable ports and legal address spaces;
* terminal memory target range and coherence-domain rules are obeyed;
* schedule, temporal tag, and resource-sharing records are consistent;
* visualization metadata references existing software or hardware
  objects and valid visualization layouts.

The verifier must not require visualization metadata. It verifies
visualization metadata only when it is present.

## Non-Goals

The mapping artifact is not a hardware architecture description. It
does not create Fabric nodes, ports, links, or domains.

The mapping artifact is not a software IR. It does not create dataflow
operations, values, or control structure.

The mapping artifact is not a GUI file format. It may carry enough
metadata to guide visualization, but a GUI remains free to choose its
own rendering algorithm.

## Acceptance Criteria

The mapping artifact target is complete when:

* one artifact can represent a full mapping from a dataflow graph to a
  Fabric system ADG;
* the artifact records thread, graph, operation, edge-route, schedule,
  buffer, memory, and resource-sharing decisions when those decisions
  exist;
* the artifact can represent both explicit and parametric thread
  binding;
* the artifact can represent temporal sharing without modifying
  software or hardware IR;
* DSE can compare multiple mapping candidates through mapping-set
  manifests;
* CGRA-sim can consume the artifact without reading PnR-internal state;
* visualization tools can render arbitrary topology and can use
  optional layout metadata for regular topologies;
* a verifier can reject stale references, illegal routes, illegal
  resource sharing, and illegal memory bindings.

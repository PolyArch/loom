# Core Dialect Boundary

This document specifies the target ownership boundary between Loom's
core dialects and tools. It is the classification rule used when a
software, hardware, mapping, runtime, or simulation fact could
otherwise be placed in more than one subsystem.

## Purpose

Loom has two primary IR sides:

* dataflow IR describes software execution and software dataflow;
* fabric IR describes physical hardware architecture.

The relation between the two is not embedded in either side. It is
recorded by the independent mapping artifact produced by PnR. Runtime
and simulators consume these artifacts according to their own
contracts, but they do not invent missing software, hardware, or
mapping facts.

The boundary rule is:

```text
Dataflow owns software semantics.
Fabric owns hardware structure.
The mapping artifact owns the chosen relation between them.
Runtime owns dynamic launch and platform execution.
DFG-sim owns dataflow-only execution evidence.
CGRA-sim owns mapped hardware-aware execution evidence.
```

## Ownership Table

| Owner | Owns | Must not own |
|-------|------|--------------|
| `dataflow.thread` | Logical software execution domains, nested logical instance structure, async thread-completion tokens, graph-launch containment legality. | Physical AccCore identity, route selection, schedule slots, temporal tags, topology coordinates, hardware capacity. |
| `dataflow.graph` | Symbol-bearing SpatialCore software DFG definitions launched from innermost executable thread bodies. | Host launch ABI, physical SpatialCore instance identity, hardware routes, resource-sharing assignments. |
| `dataflow.subgraph` | L3 software graph partitions used as candidates for FU matching and generalization. | Hardware hierarchy, PE identity, route, schedule, temporal tag, time-sharing decision. |
| `fabric.system` | System-level ADG: HostCore nodes, AccCore nodes, memories, caches, interconnect, external ports, protocol ports, directed channels, explicit one-to-one links, domains, address spaces, coherence, consistency. | Software execution semantics, selected software placement, workload schedule, simulator trace state. |
| `fabric.module` | SpatialCore or CGRA hardware templates, including PEs, FUs, switches, memories, FIFOs, boundaries, and local hardware capabilities. | A system-level SoC graph, selected workload placement, dataflow graph definition semantics. |
| Mapping artifact | Thread binding, graph binding, operation or subgraph binding, edge routes, schedule records, temporal tags, buffers, memory bindings, resource sharing, mapping diagnostics, mapping metrics. | New software operations, new hardware nodes or links, runtime fallback policy, simulator-only observations. |
| Runtime ABI | Host-visible work packages, launch descriptors, memory descriptors, launch handles, platform binding, data movement policy, synchronization policy, target selection, fallback policy, runtime diagnostics. | Dataflow token semantics, Fabric topology, PnR legality, route selection, coherence definition, FPA estimates. |
| DFG-sim | Pure dataflow execution for concrete inputs without hardware resource limits. | Fabric ADG, mapping artifacts, hardware placement, hardware routing, hardware resource contention. |
| CGRA-sim | Execution of mapped dataflow on selected hardware using the mapping artifact, with hardware resource, route, memory, buffering, timing, and activity constraints. | Choosing placements, routes, schedules, memory bindings, or resource-sharing assignments. |

## Dataflow Boundary

`dataflow.thread` is a software execution-domain carrier. Before
binding, every dynamic thread instance is logical. Selected innermost
executable thread instances may later bind to AccCore execution slots
through PnR and the mapping artifact. The dataflow verifier must not
treat a logical thread as a physical AccCore by itself.

Thread hierarchy is layered:

* non-innermost thread bodies may contain ScalarCore orchestration code
  and child `dataflow.thread.launch` operations;
* innermost executable thread bodies may contain ScalarCore residual
  code and `dataflow.graph.launch` operations;
* one direct thread-body placement level must not mix child thread
  launches and graph launches.

`!dataflow.thread_token` is the inter-thread completion token domain.
`none`-typed control values are graph, stream, memory-order, and
dataflow-control tokens. `dataflow.thread.fence` is the explicit bridge
from thread-completion and dataflow-control dependencies to a `none`
control result. `dataflow.thread.wait` consumes thread-completion
tokens for host or parent-context synchronization and produces no
graph-control value.

`dataflow.graph` is the single canonical SpatialCore software DFG
definition surface. It is symbol-bearing, function-like, module-scope,
and executes only through `dataflow.graph.launch` from an innermost
executable thread body. The target dataflow dialect has no separate
`dataflow.graph.func` surface.

`dataflow.subgraph` is software partitioning inside a `dataflow.graph`
definition. It is a candidate unit for matching or generalizing against
`fabric.fu` templates. It must not encode physical PE identity, route
identity, hardware schedule slots, temporal tags, or resource-sharing
assignments.

Logical partitioning constructs and thread-axis attributes may describe
software instance domains. They are not hardware topology. A logical
axis is not an x coordinate, y coordinate, PE coordinate, router
coordinate, route hint, or mesh statement.

## Fabric Boundary

`fabric.system` is the system-level architecture description graph. It
contains physical nodes and explicit directed connectivity. A directed
connection is from producer to consumer, output to input, master to
slave, or manager to subordinate, according to the selected protocol
schema.

`fabric.link` connects one channel endpoint to one channel endpoint.
Bundle-level helpers may exist only as syntax or builder convenience;
the verifier-visible connectivity remains directed channel endpoints.
Protocol bundle ports may contain channels with different directions,
such as AXI-MM read, write, response, and address channels. Direction
is therefore a channel property, not a single scalar property of the
whole bundle.

All hardware topology is explicit graph connectivity. Meshes,
multi-dimensional grids, x/y coordinates, and Manhattan-distance
routing are not default semantics. They may be represented by explicit
links plus optional visualization metadata. Visualization metadata may
help a GUI draw regular structures, but it must not change legality,
routing, simulation, RTL lowering, or FPA estimation.

An AccCore system node is an independent physical instance. It may
reference a `fabric.module` symbol as its SpatialCore template and may
carry ScalarCore parameters. Multiple AccCore nodes may reference the
same module symbol, but they remain distinct physical resources.

`fabric.module` is the SpatialCore or CGRA hardware template. Its local
hardware concepts, such as PE containers, FUs, switches, memories,
FIFOs, and spatial or temporal hardware capabilities, remain hardware
facts. They do not imply that software dataflow subgraphs carry
hardware time-sharing or tag semantics before mapping.

System-level selection and routing primitives must use precise names
that describe the hardware role. Deterministic one-to-one-of-many
routing uses a route decoder, contention resolution uses an arbiter,
and explicit one-to-many replication uses broadcast hardware.
System-level hardware primitives must not be named after dataflow
control operations in a way that confuses hardware routing with
`dataflow.mux` or `dataflow.demux`.

## Mapping Boundary

PnR is the only subsystem that chooses the software-to-hardware
relation. Its persistent output is the mapping artifact. PnR may use
dataflow IR, Fabric ADG, user constraints, DFG-sim reports, CGRA-sim
reports, FPA estimates, or profile data as inputs, but the chosen
relation must be serialized in the mapping artifact.

The mapping artifact records:

* which logical thread instance domains bind to which AccCore resources
  and execution batches;
* which graph launches bind to which SpatialCore execution contexts;
* which operations or subgraphs bind to which fabric resources;
* which software edges use which hardware routes;
* which schedule slots, temporal tags, buffers, queues, memory
  bindings, and resource-sharing decisions make the mapping legal.

Neither dataflow IR nor Fabric ADG is a substitute for the mapping
artifact. If a consumer needs a selected placement, route, schedule,
buffer, memory binding, or sharing decision, that fact must be in the
mapping artifact.

The detailed mapping record families are specified in
`docs/spec-mapping-artifact.md` and the related
`docs/spec-mapping-*.md` documents. PnR search policy is specified in
`docs/spec-mapping-search.md`; verification and consumer profiles are
specified in `docs/spec-mapping-verification.md`.

## Runtime Boundary

The runtime launches already compiled and mapped work. It consumes work
packages, launch descriptors, memory descriptors, mapping-artifact
identity, Fabric ADG identity, target profiles, simulator hooks, and
platform configuration.

Runtime launch handles are host-visible dynamic execution handles. They
are not `!dataflow.thread_token` values and are not `none` dataflow
control tokens.

The runtime may bind host pointers or buffer handles to
accelerator-accessible memory through platform services. This does not
add MMU or virtual-memory semantics to Fabric ADG. Fabric owns physical
memory structures, address spaces, cache coherence declarations, and
memory-consistency declarations. Runtime owns invocation-specific
memory descriptors and data movement policy.

If runtime execution needs a fact that is absent from the runtime
package, mapping artifact, Fabric ADG, dataflow artifact, target
profile, or platform configuration, the runtime must diagnose the
missing fact. It must not infer placement, routing, schedule, coherence,
fallback, or memory-binding behavior from naming conventions or
coordinates.

## Simulation Boundary

DFG-sim consumes dataflow artifacts, concrete inputs, initial memory
state, and simulator configuration. It produces dataflow-only execution
evidence and optimistic performance metrics. It does not consume Fabric
ADG or mapping artifacts.

CGRA-sim consumes dataflow artifacts, Fabric ADG, a mapping artifact,
concrete inputs, initial memory state, runtime-relevant configuration,
and simulator configuration. It produces hardware-aware execution
evidence. It may validate the mapping artifact, but it must not repair
or choose a mapping.

Comparison between DFG-sim and CGRA-sim is valid only when both reports
refer to the same workload, input data, and observable outputs. Gaps
between their metrics are acceptable only when explained by hardware
constraints that DFG-sim intentionally ignores.

## Acceptance Criteria

The core boundary is satisfied when:

* every semantic fact can be classified under exactly one primary owner;
* no verifier requires hidden side state from another subsystem to
  decide structural legality;
* dataflow IR contains no physical topology, route, PE, schedule-slot,
  temporal-tag, or hardware-capacity assumptions;
* Fabric ADG contains no selected workload placement, workload route,
  workload schedule, or software dataflow semantics;
* the mapping artifact is the only persistent source for selected
  software-to-hardware binding;
* runtime descriptors do not smuggle PnR or Fabric facts that belong in
  upstream artifacts;
* DFG-sim and CGRA-sim consume different artifact sets according to
  their contracts;
* optional visualization metadata never changes compiler, mapping,
  simulation, RTL, or FPA legality.

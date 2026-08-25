# Visualization Export

This document specifies Loom's visualization export boundary. Visualization is
a derived, removable projection of exact canonical artifacts. It is not part of
a Mapping Artifact and does not own software, hardware, Mapping, simulation,
Evaluation, or execution semantics.

## Driver Contract

`--loom-viz-export=<dir>` is a feature of the public `loom-cc` and `loom-c++`
drivers. It enables report-only capture of the artifacts needed for a
visualization projection. It does not implicitly enable acceleration, hardware
mapping, simulation, bitstream generation, or another compilation stage.

Export is best effort by default so visualization failure does not replace a
successful ordinary compiler output. A resolved strict or implementation
profile may make export failure fatal. That policy changes neither the identity
nor the semantics of the artifacts being projected.

A viewer is a consumer of the export. It is not a separate compiler, simulator,
evaluator, or Mapping authority.

## Fabric Authoring Projection

ADG Builder and builtin-target export use the same visualization owner even
when no software, Mapping, simulation, or Evaluation artifact exists. Given
one exact finalized Fabric root and an ArtifactStore, the exporter resolves
the root's exact Fabric dependency closure and emits one self-contained HTML
file beside the root's textual MLIR projection.

For a Module root, the HTML contains its complete SpatialCore resource and
directed-connectivity graph. For a System root, it contains:

* a default architecture overview that represents the NoC once, keeps every
  HostCore, AccCore, System memory service, and external boundary distinct,
  and does not promote every transport resource to an equally weighted node;
* a NoC topology view that contracts each explicit transport-resource path
  into its derived directed connectivity among architecture participants,
  while leaving service-endpoint, hardware-domain, module-attachment, and
  individual transport-resource detail to the exact architecture view;
* one architecture view with every HostCore, heterogeneous AccCore,
  InstructionCore context, SpatialCore occurrence binding, per-boundary
  endpoint attachment, memory/service endpoint,
  external boundary, transport resource, and explicit directed connection;
* one detail view for every distinct imported Module artifact, showing its
  module boundary, PEs, FUs, memories, switches, FIFOs, boundaries, typed
  ports, explicit point connections, and FU-internal configured graph; and
* exact artifact identities and typed resource summaries sufficient to
  distinguish repeated occurrences from reusable module definitions.

Repeated AccCores may reference one imported Module detail view, but every
physical AccCore occurrence remains a distinct node in the System view. The
HTML embeds no mutable hardware model and performs no ArtifactStore access.

The exporter computes every graph coordinate and edge route before writing
the HTML. Its deterministic layout handles arbitrary directed cyclic topology,
uses nested frames only for real ownership, reduces crossings, reserves routing
channels, and keeps nodes, labels, ports, edges, and ownership frames from
overlapping whenever a finite separated drawing exists. Regular construction
hints may seed this computation but never replace explicit connectivity.

Browser code may pan, zoom, fit, search, filter, inspect, and switch between
precomputed views. It must not run a force solver, graph layout engine, or
semantic topology reconstruction. The HTML is offline and self-contained: it
does not load JavaScript, CSS, fonts, icons, or data from a network location.
Selecting an architecture participant may emphasize its already-precomputed
incident NoC routes; this changes only presentation state.

## Canonical Sources

Every displayed semantic fact resolves to one exact canonical owner. The
visualization bundle records those full typed references and never selects an
implicit latest artifact.

### Program, Fabric, and Mapping Structure

The exact structural sources are:

- the Canonical Dataflow Program Artifact, `D`, for the software graph,
  operations, graph interfaces, channels, and canonical entity references;
- the Fabric Hardware Description Artifact, `F`, for hardware resources,
  endpoints, directed connectivity, clock-domain membership, and capability;
  and
- the applicable exact TechMapping, SpatialMapping, or SystemMapping Artifact
  for realization facts owned by that Mapping profile.

Placement, routes, selected resources, memory realization, tags, buffers,
event-relative resource use, transport binding, and other Mapping overlays come
only from the exact Mapping Artifact that owns them. The viewer may join those
facts with `D` and `F`; it cannot reconstruct them from a report, infer them
from screen position, or introduce an alternate mapping.

Source and compiler IR views come from their exact producing artifacts. An
export can show source, IR, and `D` when no Mapping exists. Fabric and Mapping
overlays appear only when their exact canonical artifacts are present and
compatible.

### Raw Simulation Observables

The exact `SimulationExecution` Artifact owns typed workload-execution
observations:

- terminal output values and stream sequences;
- visible logical-memory final state or diffs;
- completion and retirement observations;
- typed actor and physical-resource activity summaries.

The viewer obtains output, logical-memory, and activity from that exact
`SimulationExecution`. `loom.simulation_execution 1.0` has no persistent trace
or replay field. A viewer may additionally consume the invocation-local
`SpatialDiagnosticTrace` only when the current simulator attempt explicitly
supplies it with the exact execution context; the trace never becomes an
Execution or Evidence fact. The viewer cannot obtain normalized facts from
EvaluationEvidence, a human-readable simulator projection, a comparison projection, or
another execution with a similar case. The execution's exact Request,
observable contract, and subjects recovered through the Request determine which
Dataflow, Fabric, and Mapping objects those facts may annotate.

Activity summaries are aggregate views, not replay logs. The viewer resolves
`ActorTransitions`, `FabricResources`, and `ImplementationSignals` through
their typed owner references and exact Request lineage. It respects the
summary's progress-defined window and target-inventory coverage. For a partial
summary, a missing actor, Fabric resource, or implementation activity point is
unknown and must not be rendered as zero. Per-cycle or per-occurrence display
requires an explicitly supplied current diagnostic trace and cannot be
reconstructed from aggregates.

Architecture-only RTL or EDA checks that do not execute a workload do not
produce `SimulationExecution`. Their raw scripts, logs, and reports may be
projected from owner-attempt or scratch state, but the viewer cannot fabricate
workload outputs, memory diffs, or activity for them.

### Normalized Evaluation Results

The exact `EvaluationEvidence` Artifact is the only source for normalized:

- Evaluation outcome and typed `OutcomeReason` when applicable;
- ordinal-indexed metric results; and
- ordinal-indexed finding results and occurrences.

The viewer resolves query meaning through the exact EvaluationRequest and the
metric and finding registries. It cannot reinterpret missing findings as
absence, turn raw log text into a finding, derive a normalized metric from a
report, or apply its own pass/fail policy. Candidate ranking, quality gates,
and selection remain central DSE policy rather than visualization data.

### Reports and Bundles

`viz.bundle.json`, simulation projections, comparison projections, FPA summaries,
generated HTML, JavaScript, CSS, indexes, and renderer state are projections
only. They may reference or cache canonical facts for efficient presentation,
but they are never semantic inputs to another projection and never become
schema authorities.

Visualization bundle version 1.1 adds removable resource-time evidence. The
exporter first verifies the compiler-built finite transition graph, including
entry reachability and every exact Mapping/Deployment endpoint, then replays
each transition closure and projects trigger, safe point, active allocations,
owner-derived deltas and costs, and parent/child spectrum summaries. The
endpoint array is derived from the graph rather than retained as another
catalog. An empty array means the application produced no verified finite
edge; visualization never upgrades an incomplete edge or supplies runtime
selection semantics.

Scripts, stdout, stderr, vendor warnings, tool-native reports, diagnostic
traces, and other raw execution material remain attempt or scratch material
associated with the exact Request. No current raw-bundle Artifact or trace
chunk schema is implied. These records cannot
replace `SimulationExecution` for workload observables or
EvaluationEvidence for normalized outcome, metrics, and findings.

## Exact Projection Joins

An export is a set of verified joins over typed references, not a directory
name convention. Before presenting an overlay, the exporter verifies:

- every referenced Artifact schema and identity;
- the dependency closure among `D`, `F`, and the exact Mapping Artifact;
- the `SimulationExecution` exact Request reference and the workload
  observable contract recovered through it;
- any trace event's actor, resource, and Mapping context references; and
- the EvaluationEvidence reference to its exact Request and typed output
  bindings.

Facts from distinct requests, mappings, model bindings, or simulation
executions remain visibly distinct even if their source text or display labels
match. A malformed or incompatible join is an export failure. It is not a
reason to reinterpret or mutate a canonical artifact.

Cached copies are disposable. Rebuilding from the same canonical inputs and
resolved visualization configuration reproduces the same semantic references.
Interactive camera, filtering, selection, and visibility state can vary
without changing any artifact or semantic identity.

## Diagnostic Activity Replay

Persistent activity replay is unavailable. When the current attempt explicitly
supplies a `SpatialDiagnosticTrace`, the viewer may project only its typed
frames and events. Timed SpatialCore diagnostics use the `EventCoordinate`
owned by `docs/spec-simulation-artifacts.md`; visualization does not redefine
that reference.

`reference_cycle` is always a canonical `ExactRatio`; an integral cycle `N`
has the sole canonical form `N/1`. `delta` expresses same-cycle causal
propagation and is not another cycle or a latency metric. Trace frames are
strictly increasing by EventCoordinate. Events within one frame use a stable
typed canonical serialization key; that ordering does not invent arbitration
or execution semantics.

The viewer recovers launch, graph-retirement, and terminal markers from the
execution's `SpatialProgressObservations`. It does not infer them from trace
events, duplicate elapsed cycles, or choose a clock domain. The exact DFG model
or the exact Fabric, SpatialMapping, and mapped launch boundary own the
reference-domain meaning.

An actor firing is shown from `ActorCommitted`; `ActorRetired` may occur at a
later coordinate. `TokenPublished` supplies the exact semantic token shown on
an endpoint, and `MemoryLinearized` supplies only the primitive dynamic memory
relations selected by the diagnostic contract. Each record resolves through its
execution-local graph invocation, actor transition, token, or memory-action
occurrence to the exact Dataflow owner.

A microarchitecture replay projects `PhysicalRequested`,
`PhysicalGranted`, and `PhysicalRetired` onto the exact Fabric use pattern or
selected traversal set named by the event. The request-to-grant interval is
shown as stall time. Queue, occupancy, resource-state, Tag, and configuration
views are derived from Fabric, Mapping, and this lifecycle; the viewer cannot
consume a second event vocabulary or persist derived deltas as another
authority.

Replay is an activity projection, not a checkpoint or complete state-replay
log. It must not copy simulator queues, token state, or logical memory into a
second authority. Final outputs, streams, and logical-memory state or diffs
remain fields of the exact `SimulationExecution`; metrics, findings, and
Evaluation outcome remain fields of exact EvaluationEvidence.

Diagnostic capture level affects execution cost and scratch material, not
simulation behavior. The viewer respects the supplied frame and event order,
never fills missing cycles or events from a report, and never interprets an
absent or empty diagnostic value as a complete execution history.

## Mapping and Layout Exclusion

TechMapping, SpatialMapping, and SystemMapping are the only Mapping profiles.
There is no visualization Mapping profile.

Layout records, view definitions, styles, labels, groups, overlays, metric
display hints, and GUI payloads do not participate in Mapping identity,
completeness, verification, legality, ranking, simulation, runtime,
configuration generation, RTL lowering, or Evaluation. Deleting the complete
visualization export changes none of those results.

Every valid Fabric topology is renderable from explicit resources, endpoints,
and directed connectivity. The renderer supports an arbitrary-topology graph
without coordinates or regular-layout metadata. It never infers connectivity,
legal placement, routing adjacency, distance, or cost from visual position.

Regular-construction helpers may provide an authoring-only, nonsemantic
`visual_layout` hint so meshes, arrays, rings, pipelines, or stacked grids have
a familiar presentation. Fabric finalization strips the hint before canonical
semantic serialization and identity generation. When retained, the exporter
stores it only in a removable visualization projection that references the
exact finalized Fabric identity. A hint cannot create Fabric edges, constrain
Mapping, select routes, define schedules, or change legality. If a hint is
absent or invalid, the renderer falls back to the explicit graph rather than
inventing semantic adjacency. Viewer-selected layout and state are not written
back into Fabric or Mapping.

## Conformance Anchor

Only this stable semantic anchor belongs at this boundary:

- Given exact `D`, `F`, Mapping, and `SimulationExecution` references, activity
  views resolve actor and physical-resource summaries to those exact artifacts,
  preserve exact summary window and coverage semantics, and obtain no execution
  or Evaluation fact from report or UI state. When the same current attempt
  explicitly supplies an invocation-local `SpatialDiagnosticTrace`, replay
  renders only its typed canonically ordered frames, claims no coverage, and
  leaves Artifact identity unchanged.
- Given an exact System Fabric root and its published imported Modules, a
  self-contained export displays every AccCore occurrence and every distinct
  SpatialCore topology using statically computed geometry, without browser-side
  graph layout or a second hardware fact owner.

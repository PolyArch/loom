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
- typed actor and physical-resource activity summaries; and
- the typed trace manifest, including ordered content-addressed chunk
  references, coverage, and completeness.

The viewer obtains output, logical-memory, activity, and trace ordering from
that exact `SimulationExecution`; it resolves opaque chunk payloads from the
manifest's exact raw detailed bundle. It cannot obtain normalized facts from
EvaluationEvidence, a human-readable simulator projection, a comparison projection, or
another execution with a similar case. The execution's exact Request,
observable contract, and subjects recovered through the Request determine which
Dataflow, Fabric, and Mapping objects those facts may annotate.

Architecture-only RTL or EDA checks that do not execute a workload do not
produce `SimulationExecution`. Their raw scripts, logs, and reports may be
projected from exact detailed-bundle and owner-attempt references, but the
viewer cannot fabricate workload outputs, memory diffs, or activity for them.

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

A raw detailed bundle may provide scripts, stdout, stderr, vendor warnings,
tool-native reports, opaque trace chunks, and other execution material. Those
records remain raw material associated with the exact Request. They cannot
replace `SimulationExecution` for workload observables or
EvaluationEvidence for normalized outcome, metrics, and findings.

## Exact Projection Joins

An export is a set of verified joins over typed references, not a directory
name convention. Before presenting an overlay, the exporter verifies:

- every referenced Artifact schema and identity;
- the dependency closure among `D`, `F`, and the exact Mapping Artifact;
- the `SimulationExecution` Request and observable-contract references;
- any trace event's actor, resource, and Mapping context references; and
- the EvaluationEvidence reference to its exact Request and retained detailed
  material.

Facts from distinct requests, mappings, model bindings, or simulation
executions remain visibly distinct even if their source text or display labels
match. A malformed or incompatible join is an export failure. It is not a
reason to reinterpret or mutate a canonical artifact.

Cached copies are disposable. Rebuilding from the same canonical inputs and
resolved visualization configuration reproduces the same semantic references.
Interactive camera, filtering, selection, and visibility state can vary
without changing any artifact or semantic identity.

## Activity Replay

Activity replay uses the exact trace manifest owned by `SimulationExecution`
and the opaque chunks it orders from the referenced detailed bundle.
Timed SpatialCore traces use:

```text
EventCoordinate = (reference_cycle, delta)
```

`reference_cycle` is a nonnegative integer or canonical `ExactRatio` cycle.
`delta` expresses same-cycle causal propagation and is not another cycle or a
latency metric. Trace frames are strictly increasing by EventCoordinate. Events
within one frame use a stable typed canonical serialization key; that ordering
does not invent arbitration or execution semantics.

An actor firing is shown at its commit coordinate. Commit and retirement may
occur at different coordinates. Each firing record resolves its canonical
actor entity, execution-local invocation occurrence, per-actor firing ordinal,
and EventCoordinate. A microarchitecture trace may additionally reference the
exact Fabric resource and Mapping context responsible for route, buffer,
resource, or stall activity.

Replay is an activity projection, not a checkpoint or complete state-replay
log. It must not copy simulator queues, token state, or logical memory into a
second authority. Final outputs, streams, and logical-memory state or diffs
remain fields of the exact `SimulationExecution`; metrics, findings, and
Evaluation outcome remain fields of exact EvaluationEvidence.

Trace capture level and chunking affect retained raw material and execution
cost, not simulation behavior. The viewer respects the trace manifest's order
and completeness and never fills missing cycles or events from a report.

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
  replay follows strictly cycle-ordered trace frames and stable same-frame
  event order, resolves actor and physical-resource activity to those exact
  artifacts, and obtains no execution or Evaluation fact from report or UI
  state.

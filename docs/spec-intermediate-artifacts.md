# Intermediate Reports And Projections

This document defines the boundary for nonsemantic reports emitted while a
Loom invocation is running. Reports are human-facing or tool-convenience
projections of canonical artifacts and owner-specific execution records. They
are not compilation, Mapping, simulation, Evaluation, deployment, or hardware
authorities.

## Ownership

Every report field that describes a semantic fact resolves to one exact owner:

- a Canonical Dataflow Program owns software structure and behavior;
- a Fabric Hardware Description owns hardware structure and capability;
- a complete Mapping Artifact owns selected realization, placement, routes,
  resource use, tags, buffers, and other profile-specific decisions;
- a `SimulationExecution` owns typed terminal workload observations, activity
  summaries, and the trace manifest;
- an `EvaluationRequest` owns the exact evaluation question;
- an `EvaluationEvidence` owns normalized outcome, metric results, and finding
  results;
- a detailed bundle owns retained scripts, logs, raw tool reports, canonical
  trace chunks, and other payloads; and
- `InvocationManifest`, `ExecutionJournal`, and owner-specific attempt records
  own invocation provenance, recovery state, and retry history.

A report may reference and render those facts. It cannot copy a selected
Mapping relation, reinterpret a trace, normalize a metric, infer a missing
finding, or publish an invocation status as another semantic result.

## Projection Contract

Each report declares its projection schema and exact source references. A
schema version uses `X.Y`, where `X` denotes an incompatible schema change and
`Y` denotes a compatible extension. Report schema versioning does not create
another version space for any source artifact.

Projection generation verifies every source identity and cross-artifact join.
A missing, malformed, or incompatible source produces a report-generation
diagnostic; the producer must not substitute a nearby file, an implicit latest
artifact, or a report with a matching display label.

Reports may contain summaries, formatting choices, display labels, and
navigation indexes. Cached source fragments are disposable and must retain
their exact owner references. Removing every report leaves compilation,
Mapping, simulation, Evaluation, deployment, runtime, and hardware behavior
unchanged.

Filesystem paths, output directories, compression, and presentation format
are invocation bindings. They do not enter source artifact identity. Consumers
must receive report paths explicitly and must not discover semantic inputs by
scanning a scratch directory.

## Simulation Reports

A DFG-sim, CGRA-sim, or system-simulation report obtains terminal values,
stream sequences, visible logical-memory state or diffs, completion, activity,
and trace order and coverage only from the exact `SimulationExecution`. It
resolves canonical trace chunks only through the manifest's exact detailed-
bundle reference. It obtains normalized outcome, metrics, and findings only
from exact `EvaluationEvidence`, with query meaning recovered through the
corresponding `EvaluationRequest` and registries.

For a present trace, the report resolves the manifest's one exact same-Request
detailed bundle, verifies every ordered `BlobDigest`, and decodes canonical
chunk bytes. It preserves `Complete` versus launch-rooted `Prefix` coverage and
must not invent a late-start range, fill an interior gap, reorder chunks, or
treat an absent manifest as an empty complete trace.

A simulator progress counter, event count, raw tool exit status, or
human-oriented score is not a cycle metric unless an Evaluation model has
produced the corresponding typed metric result. Invocation-local memory
handles and diagnostic labels are not promoted to persistent logical-memory
identity.

Architecture-only RTL or EDA evaluation has no workload execution and
therefore produces no empty `SimulationExecution`. Its report projects exact
Request, Evidence, detailed-bundle, and owner-attempt records instead.

## Mapping Reports

The only canonical persistent Mapping outputs are the complete roots defined
by `docs/spec-mapping-artifact.md`. Mutable candidates, `FrozenModel`, closure
projections, dense indices, caches, configured views, and runtime images are
nonpersistent implementation state and cannot be published as Mapping
substitutes.

A successful Mapping report references the exact finalized Mapping identity
and may summarize its invocation. It cannot duplicate selected records or
become an alternative input. Invalid input, infeasibility, unsupported proof,
constraint rejection, or budget exhaustion remains a typed invocation result;
it is never encoded as a partial Mapping Artifact.

## Visualization

JSON, HTML, JavaScript, CSS, indexes, and renderer state used by
`--loom-viz-export` follow `docs/spec-mapping-visualization.md`. They are
removable projections and never become a Mapping profile or schema authority.

## Conformance Anchor

Given exact source references, regenerating a report may change presentation
but must preserve every referenced semantic fact. No report is accepted where
the corresponding canonical artifact, execution, Evidence, or owner record is
required.

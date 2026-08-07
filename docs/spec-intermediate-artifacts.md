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
- a `SimulationExecution` 1.0 owns typed terminal workload observations and
  activity summaries but no trace field;
- an `EvaluationRequest` owns the exact evaluation question;
- an `EvaluationEvidence` owns normalized outcome, metric results, and finding
  results;
- owner-attempt or scratch storage retains ExternalToolInvocationBundles,
  scripts, logs, raw tool reports, invocation-local diagnostic traces, and
  other nonsemantic payloads; and
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

Direct EDA projections, their source Evidence, and their owner-attempt material
are local-only under the repository disclosure boundary in
[EDA Tooling](spec-eda-tooling.md). Normalization into Evidence does not make
captured EDA data eligible for repository tracking.

## Simulation Reports

A DFG-sim, CGRA-sim, or system-simulation report obtains terminal values,
stream sequences, visible logical-memory state or diffs, completion, and
activity only from the exact `SimulationExecution`. It obtains normalized
outcome, metrics, and findings only from exact `EvaluationEvidence`, with query
meaning recovered through the corresponding `EvaluationRequest` and
registries. Persistent trace projection is unavailable in
`loom.simulation_execution 1.0`; an invocation-local diagnostic trace may be
projected only when the current attempt explicitly supplies it.

A simulator progress counter, event count, raw tool exit status, or
human-oriented score is not a cycle metric unless an Evaluation model has
produced the corresponding typed metric result. Invocation-local memory
handles and diagnostic labels are not promoted to persistent logical-memory
identity.

Architecture-only RTL or EDA evaluation has no workload execution and
therefore produces no empty `SimulationExecution`. Its report projects exact
Request, Evidence, and owner-attempt records instead. Raw payloads remain
scratch state until their exact Artifact owner exists.

An external-tool report may reference the exact bundle and completion record,
but it cannot discover outputs by scanning the bundle directory. The provider
importer first validates the declared output inventory and produces typed
Evidence; human reports project that result rather than normalize vendor text
again.

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

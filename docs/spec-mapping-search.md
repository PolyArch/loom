# Mapping Search and PnR Policy

This document specifies the PnR search policy that produces Loom
mapping artifacts. The output data model is specified by
`docs/spec-mapping-artifact.md` and the detailed
`docs/spec-mapping-*.md` record-family specs.

PnR is a mapper from software abstractions to hardware abstractions. It
does not simulate execution, mutate dataflow IR, mutate Fabric ADG, or
invent missing hardware resources.

PnR is an NP-hard placement and routing problem. The deterministic
baseline described below is a correctness oracle and debugging policy,
not the final performance policy. Production-quality policies must be
designed for large candidate spaces, cache-local data structures,
incremental legality updates, and bounded-time search.

The first hard target for this spec is artifact quality, not search
cleverness: Loom must be able to validate, consume, and emit complete
mapping artifacts, including manually produced fixtures. Scalable search
engines are search-extension targets layered on top of the same artifact
and verifier contracts.

The central compute mapping relation is `dataflow.subgraph` to
`fabric.fu`. Operation-level binding is allowed when the software graph
has not been partitioned or when a consumer explicitly requests that
granularity, but the SpatialCore CGRA target is the FU-compatible
software subgraph. PE, switch, memory, and boundary records exist to
make that subgraph-to-FU mapping executable on the selected hardware.

## Search Inputs

Required inputs:

* software dataflow IR after compiler placement boundaries are chosen;
* selected `fabric.system`;
* every referenced `fabric.module` template;
* PnR option set;
* workload shape when the mapping is shape-dependent.

Optional inputs:

* user constraints;
* DFG-sim reports;
* prior CGRA-sim reports;
* FPA reports;
* prior mapping-set manifests;
* visualization preferences.

Optional inputs may influence cost or pruning. They must not create
legality facts that are absent from software IR, Fabric ADG, or the
mapping artifact being built.

## Candidate State

PnR-internal candidate state may contain mutable solver data, priority
queues, backtracking frames, congestion estimates, and rejected partial
candidates. None of that internal state is a valid persistent artifact.

A candidate becomes persistent only when serialized into a mapping
artifact that contains all records required by the relevant consumer
profile in `docs/spec-mapping-verification.md`.

## Legality Pipeline

Every PnR policy uses the same legality pipeline:

1. identity and reference legality;
2. thread placement legality;
3. graph placement legality;
4. operation/subgraph placement legality;
5. route legality;
6. schedule, temporal-tag, reconfiguration, and resource-sharing
   legality;
7. buffer legality;
8. memory, coherence, consistency, and memory-order legality;
9. consumer-profile completeness.

The cost model may rank only candidates that pass the legality rules
needed for the requested output profile. A cost model must never make an
illegal candidate legal.

## Arbitrary-Topology Rule

PnR must treat hardware as an arbitrary directed graph.

Placement candidates are generated from explicit hardware nodes and
resources. Route candidates are generated from explicit directed channel
endpoints, `fabric.link` connectivity, module resources, boundaries,
adapters, FIFOs, buffers, and declared protocol channels.

Coordinates, grid metadata, layout metadata, labels, and visualization
positions are never topology. They may affect visualization only. Cost
models use explicit hardware weights such as latency, bandwidth,
capacity, or user-declared edge weights; they must not derive hardware
cost from visualization coordinates.

## Deterministic Baseline

The required baseline policy is deterministic and debug-friendly.

Candidate construction order:

1. process thread instance domains in stable logical order;
2. process graph launches in stable software order;
3. process operation and subgraph units in dependency-topological order,
   with stable symbolic tie breakers;
4. enumerate compatible hardware resources in stable symbol order;
5. enumerate route paths over the explicit hardware graph using a stable
   shortest-legal-path metric;
6. assign earliest legal schedule/resource-use records;
7. assign required buffers and memory bindings;
8. serialize records in deterministic artifact order.

The baseline policy is not required to be performance-optimal. Its job
is to be correct, reproducible, diagnosable, and useful as a reference
for tests.

## Search Extensions

Loom may implement additional PnR policies:

* beam search;
* simulated annealing;
* integer or mixed-integer programming;
* min-cost flow or multi-commodity flow routing;
* improved A* routing over explicit directed channel endpoints;
* profile-guided search;
* feedback-driven DSE using CGRA-sim or FPA reports;
* user-guided constrained search.

Every policy must emit the same artifact schema and pass the same
verifier profiles. Search-policy-specific state belongs in logs or
mapping-set manifests, not in per-candidate mapping records.

## Cost Model

The PnR cost model ranks legal candidates. Required baseline terms:

* unmapped required software objects;
* route length and route resource pressure;
* exclusive resource pressure;
* PE-local FU activation pressure;
* buffer depth and buffer pressure;
* schedule length or estimated cycles;
* memory bandwidth and coherence pressure;
* temporal tag pressure and tag-capacity pressure;
* reconfiguration count;
* cache locality of PnR solver data structures and incremental search
  updates when comparing otherwise equivalent policies;
* diagnostics severity;
* optional DFG-sim, CGRA-sim, and FPA feedback references.

The baseline cost model must define a total deterministic order.
Additional policies may use weighted objectives, lexicographic
objectives, constraints plus objectives, or Pareto ranking. The chosen
cost configuration must be recorded in the mapping artifact metrics or
mapping-set manifest.

## Diagnostics During Search

PnR must emit structured diagnostics for:

* no compatible AccCore;
* no compatible SpatialCore module;
* no compatible FU/PE/memory resource;
* no legal route;
* route resource contention;
* schedule/resource conflict;
* insufficient buffer resources;
* unsupported memory/coherence/consistency requirement;
* missing consumer-profile records.

Diagnostics should be attached to the most specific software object and
hardware object known at the time. Rejected internal candidates do not
need full artifact records, but user-visible failures must be
represented by diagnostic records or reports.

## Mapping-Set Manifest

A DSE run that produces multiple candidates emits a mapping-set
manifest. Required fields:

* shared software input references;
* shared hardware input references;
* PnR policies and option sets;
* objective functions;
* candidate artifact list;
* rejected-candidate summaries;
* selected candidate or Pareto set;
* summary metrics and report references.

The manifest must not duplicate detailed placement, route, schedule,
buffer, or memory records. Those remain in per-candidate artifacts.

## Validation

Search-policy tests must include:

* deterministic baseline on arbitrary non-mesh topology;
* deterministic baseline on mesh-like topology using explicit links;
* subgraph-to-FU binding as the primary compute-placement case;
* PE with multiple candidate FUs where only one FU can be active for a
  spatial or temporal resource-use slot;
* negative no-route case;
* negative incompatible-resource case;
* resource-conflict case requiring schedule or tag records;
* boundary tag assignment that fails because the required tag value
  cannot be represented by the hardware tag width;
* memory/coherence negative case;
* at least one multi-candidate mapping-set manifest;
* replay test proving the selected candidate is reproducible from the
  recorded policy and inputs.

## Acceptance Criteria

Mapping search is complete when:

* PnR can produce a verifier-clean artifact for a non-mesh hardware
  graph;
* PnR can produce a verifier-clean artifact for a regular mesh-like
  hardware graph without using coordinates as connectivity;
* every legal candidate selected by any policy can be serialized into
  detailed mapping records;
* every failed required mapping emits structured diagnostics;
* DSE can compare multiple candidate artifacts through a mapping-set
  manifest;
* scalable policies such as simulated-annealing-style placement or
  improved-A*-style routing can be added without changing the artifact
  schema or weakening the deterministic baseline as a reference oracle;
* CGRA-sim can consume selected artifacts without PnR internal state.

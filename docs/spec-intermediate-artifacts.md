# Intermediate Artifact Formats

This document describes the intermediate report formats emitted by the
implemented Loom mapping and simulation tools. It does not define future RTL,
EDA, design-space exploration, or report-bundle formats.

## Common Conventions

JSON reports identify their format with `kind` and record a `status`. A passing
report contains the evidence required by its producer. A non-passing report
contains diagnostics instead of fabricated numeric values. Producers use a
subset of this accepted interoperability vocabulary:

Schema versions are strings in `X.Y` form. `X` changes are
breaking/incompatible and `Y` changes are non-breaking.

The currently used status values are:

* `pass`: the requested operation completed with usable evidence;
* `fail`: the operation ran but did not produce a valid result;
* `unsupported`: the input is outside the implemented scope;
* `skipped`: policy intentionally omitted the operation;
* `blocked`: a required input or prerequisite is unavailable;
* `not_run`: the operation was not attempted.

Artifact paths are explicit command inputs and outputs. Consumers must not
discover reports by scanning nearby scratch directories.

## DFG Simulation Report

`loom-dfg-sim` emits a JSON object with `schema_version = "2.2"` and
`kind = dfg_sim_report`.

Representative fields:

* `schema_version`;
* `kind`;
* `workload`;
* `graph`;
* `status`;
* `metric_definition`;
* `operation_semantics_source`;
* `operation_cost_model_source`;
* `operation_cost_score`;
* `weighted_operation_score`;
* `modeled_library_score`;
* `operation_diversity_score`;
* `memory_address_score`;
* `score_breakdown`;
* `dynamic_work_items`;
* `operation_fire_counts`;
* `modeled_library_calls`;
* `final_outputs`;
* `final_memory_state`;
* `final_memory_roots`;
* `diagnostics`.

Scores, dynamic work counts, and operation fire counts are non-negative.
`operation_cost_score` is the sum of weighted operation fires, modeled library
work, operation diversity, and computed-memory-address scores. The metric
definition is
`weighted_operations_plus_library_work_diversity_and_address.v1`. This is a
deterministic heuristic, not a timing model. `wavefront_steps` and
`event_count` describe simulator progress and activity; neither field is a
cycle estimate. Final outputs and visible memory state use deterministic
encodings for functional regression checks. `final_memory_roots` maps each
imported or exported memory port to a derived invocation-local alias-class
label, so aliases within the reported invocation are observable without
treating a memory capability as scalar payload. These labels are not stable
memory-object identities and may be reused across invocations. Stable
cross-invocation identity is unimplemented and remains a blocker for consumers
that need to correlate memory objects over time.

## PnR Publication Boundary

The legacy graph-to-Fabric rematcher and its JSON mapping/estimate tools are
not canonical artifact producers or consumers. They have been removed rather
than retained as a second Mapping authority.

The implemented PnR boundary is currently the native C++ Mapping verifier plus
`FrozenRealizationGraph` and `FrozenRoutingGraph`. A developer or product CLI
must wait for the dedicated Mapping MLIR persistence layer and the resolved
PnR Config, search, and persistent SpatialMapping schema. Exact SpatialMapping
records remain open. JSON may be emitted later as a reporting or visualization
projection, but it is not a canonical Mapping input.

## Scope

These formats cover only currently connected simulation boundaries. RTL
manifests, EDA reports, whole-stack audit summaries, and DSE report bundles
need separate formats when their real producers and consumers exist.

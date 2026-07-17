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
imported or exported memory port to its derived logical root label, so aliases
are observable without treating a memory capability as scalar payload.

## PnR Mapping Outputs

`loom-pnr-map` consumes one `dataflow.graph` and one Fabric hardware root.
It emits a compact CSV row and can also emit a JSON mapping artifact.

The CSV columns are:

```text
workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
```

The JSON artifact has `schema_version = "2.0"` and `kind = pnr_mapping`.
Representative fields include:

* workload, graph, and hardware identities;
* mapping and configuration identities;
* placement, routed-edge, unrouted-edge, and configuration counts;
* `placements`;
* `routes`;
* `unrouted_edge_details`;
* `config_bitstream`;
* `status` and `diagnostics`.

A passing mapping has no unplaced software records or unrouted edges. Counts
must agree with their corresponding arrays. Route segments carry concrete
endpoints consumed by mapping validation. `resource_pressure`, when present, is
diagnostic metadata rather than a stable pass/fail policy.

Consumers require the canonical `"1.0"` string.

## Mapping Estimate Report

`loom-mapping-estimate` consumes a PnR mapping artifact. Fabric MLIR is an
optional validation input. The tool emits a JSON object with
`schema_version = "1.0"` and `kind = mapping_estimate_report`.

Representative fields include:

* workload, hardware, and mapping identities;
* `status` and `diagnostics`;
* placement, route, configuration, and schedule counts;
* weighted component scores and `total_cost_score`;
* `score_breakdown` and `limitations`.

The score is a deterministic heuristic derived from mapping artifact counts.
It is neither a timing model nor a functional simulator and does not report
cycle counts or program outputs.

## Scope

These formats cover only currently connected mapping and simulation
boundaries. RTL manifests, EDA reports, whole-stack audit summaries, and DSE
report bundles need separate formats when their real producers and consumers
exist.

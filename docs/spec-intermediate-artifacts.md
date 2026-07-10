# Intermediate Artifact Formats

This document describes the intermediate report formats emitted by the
implemented Loom mapping and simulation tools. It does not define future RTL,
EDA, design-space exploration, or report-bundle formats.

## Common Conventions

JSON reports identify their format with `kind` and record a `status`. A passing
report contains the evidence required by its producer. A non-passing report
contains diagnostics instead of fabricated numeric values. Producers use a
subset of this accepted interoperability vocabulary:

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

`loom-dfg-sim` emits a JSON object with `kind = dfg_sim_report`.

Representative fields:

* `schema_version`;
* `kind`;
* `workload`;
* `graph`;
* `status`;
* `operation_semantics_source`;
* `operation_cost_model_source`;
* `optimistic_cycles`;
* `dynamic_work_items`;
* `operation_fire_counts`;
* `final_outputs`;
* `final_memory_state`;
* `diagnostics`.

Cycle counts, dynamic work counts, and operation fire counts are non-negative.
Final outputs and visible memory state use deterministic encodings so a later
hardware-aware simulation can compare functional results.

## PnR Mapping Outputs

`loom-pnr-map` consumes one `dataflow.graph.func` and one Fabric hardware root.
It emits a compact CSV row and can also emit a JSON mapping artifact.

The CSV columns are:

```text
workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
```

The JSON artifact has `kind = pnr_mapping`. Representative fields include:

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
endpoints consumed by CGRA simulation. `resource_pressure`, when present, is
diagnostic metadata rather than a stable pass/fail policy.

## CGRA Simulation Report

`loom-cgra-sim` consumes a DFG simulation report and a PnR mapping artifact.
Fabric MLIR is an optional validation input. The tool emits a JSON object with
`kind = cgra_sim_report`.

Representative fields include:

* workload, hardware, and mapping identities;
* `status` and `diagnostics`;
* operation semantics and cost-model identities;
* `dfg_cycles`;
* `hardware_aware_cycles`;
* `performance_delta_cycles`;
* modeled routing, memory, width-adapter, contention, and scheduling costs;
* `final_outputs`;
* `final_memory_state`;
* `functional_state_source`.

For comparable passing reports, hardware-aware cycles are not smaller than DFG
cycles. The reported performance delta agrees with the modeled cost
components. Functional state is carried from the DFG simulation evidence after
the report fields are validated.

## Simulator Cycle Summary

`loom-sim-cycle-summary` projects explicit DFG and optional CGRA report paths
into CSV. It requires at least one `--dfg-report` argument.

The CSV columns are:

```text
kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
```

Supplied reports for the same workload are aggregated by summing their cycle
counts. If any supplied report is incomplete, the aggregate remains
non-passing and does not expose a partial total. A passing CGRA total must not
be smaller than the corresponding DFG total.

## Scope

These formats cover only currently connected mapping and simulation
boundaries. RTL manifests, EDA reports, whole-stack audit summaries, and DSE
report bundles need separate formats when their real producers and consumers
exist.

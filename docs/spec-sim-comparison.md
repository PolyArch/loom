# Simulation Comparison

Implementation status: this document is a target contract. Loom does not
currently build the CGRA simulator or comparison producer required by this
contract.

This document specifies how Loom compares DFG-sim and CGRA-sim results.
DFG-sim is specified in `docs/spec-sim-dfg.md`. CGRA-sim is specified
in `docs/spec-sim-cgra.md`.

## Purpose

The comparison protocol answers this question:

```text
For the same workload and input, do pure dataflow semantics and mapped
hardware-aware behavior agree, and are performance differences
explained by hardware constraints?
```

The protocol consumes:

* a DFG-sim report;
* a CGRA-sim report;
* optional mapping artifact metadata;
* optional comparison configuration.

It produces:

* functional comparison status;
* memory-state comparison status;
* performance comparison status;
* explanation categories for differences;
* diagnostics;
* a comparison report usable by tests, DSE, and user-facing reports;
* optional comparison table exports for regression dashboards.

## Shared Identity

Reports are comparable only when they refer to the same workload
identity and runtime input identity. When fingerprints are present, the
comparison tool must check them. A mismatch is a comparison diagnostic,
not a performance result.

The CGRA-sim report must also identify the mapping artifact it consumed.
The DFG-sim report does not need a mapping artifact because DFG-sim
does not consume hardware mapping.

## Functional Comparison

For a legal mapping and supported simulator scope, DFG-sim and CGRA-sim
must produce equivalent functional outputs and equivalent final memory
state for visible memory regions.

The comparison must be based on final outputs and final memory diffs
when both simulator runs pass and the modeled scope is supported. A
final-state comparison is valid only when both reports define the same
visible memory region set or provide enough metadata to derive the same
diff region set.

Functional mismatch is a correctness failure unless explicitly explained
by unsupported operations, intentionally partial simulation scope, or
invalid mapping diagnostics.

## Performance Comparison

DFG-sim reports software activity counts and heuristic scores. CGRA-sim
reports hardware-aware metrics. These metrics are not automatically the same
unit and must not be presented as a cycle ratio.

If a future DFG timing model and CGRA-sim expose comparable timing metrics,
the comparison must first verify their metric definitions and units. A
comparison tool must classify unmatched definitions as
`metric_not_comparable` instead of converting activity scores into cycles.

Expected CGRA-sim overhead categories include:

* finite PE, FU, memory, buffer, or route resources;
* route latency or congestion;
* memory bandwidth, cache, coherence, or consistency constraints;
* backpressure and queue occupancy limits;
* temporal sharing and temporal tags;
* schedule slots and reconfiguration;
* clock-domain crossing costs;
* protocol conversion, width conversion, arbitration, or broadcast
  costs;
* modeled ScalarCore residual execution;
* simulator fidelity settings.

DFG-sim may provide functional and software-activity baselines. Performance
ratios require explicit, compatible metric definitions from both producers.

## Difference Classification

The comparison report classifies each difference as one of:

* `match`: no meaningful difference;
* `expected_hardware_constraint`: difference explained by hardware
  limits that DFG-sim ignores;
* `metric_not_comparable`: reports use different metric definitions;
* `unsupported_scope`: at least one simulator did not model required
  behavior;
* `mapping_invalid`: CGRA-sim rejected or diagnosed the mapping;
* `functional_mismatch`: outputs or visible memory state differ without
  an accepted explanation;
* `report_mismatch`: workload identity, input identity, schema, or
  fingerprint does not match.

`functional_mismatch` and `report_mismatch` are failing outcomes for
tests unless the test explicitly expects the diagnostic.

## Report Contract

A simulation comparison report must include:

* DFG-sim report identity;
* CGRA-sim report identity;
* mapping artifact identity when present;
* workload and input identity;
* functional comparison result;
* memory comparison result;
* visible memory region set and diff derivation when memory comparison
  is performed;
* blocked or unsupported-scope diagnostics when either simulator report
  lacks required final outputs or final memory state;
* performance metric definitions;
* performance comparison result;
* difference classification;
* diagnostics and explanation categories.

## Metric Table Export

A future comparison tool may emit a compact table for regression tracking
only after both simulator report schemas define the projected metrics. Every
numeric column must identify its metric definition and unit. Workload, input,
mapping, classification, and diagnostics columns may provide context.

If either simulator report is unavailable or its metric is not comparable,
the row must contain an explicit status or diagnostic marker. Missing evidence
must not be represented as numeric zero. Aggregation across graph slices is
legal only when source reports share workload, input, metric definition, and
unit identities.

## Use By PnR And DSE

PnR may consume comparison reports from previous candidates as cost
feedback. A comparison report is evidence for a later search decision;
it is not part of the original PnR decision unless a mapping-set
manifest explicitly references it.

DSE may use comparison reports to reject candidates, choose objective
weights, or identify hardware bottlenecks.

## Acceptance Criteria

The comparison protocol is complete at the target-spec level when:

* it verifies that DFG-sim and CGRA-sim reports refer to the same
  workload and input;
* it detects functional output and visible-memory-diff mismatches;
* it distinguishes correctness mismatches from expected hardware
  constraint differences;
* it reports metric definitions before performance ratios;
* it explains which hardware constraints affect CGRA-sim metrics;
* it can feed PnR and DSE as explicit evidence;
* any table export records metric definitions, units, and missing-evidence
  status explicitly.

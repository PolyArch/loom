# Simulation Comparison

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

DFG-sim reports optimistic software metrics. CGRA-sim reports
hardware-aware metrics. These metrics are not automatically the same
unit.

When both reports expose a comparable cycle or step estimate,
CGRA-sim's constrained cycle count should be no more optimistic than
DFG-sim's unconstrained estimate. If CGRA-sim reports a lower number,
the comparison report must explain why the numbers are not comparable or
which modeling assumption caused the inversion.

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

DFG-sim may be used as a lower-bound-style reference, but the comparison
tool must report the metric definitions before presenting ratios.

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

## Cycle Table Export

The comparison tool must be able to emit a compact cycle table for
mid-level regression tracking. The canonical table form has one row per
kernel or app case and these required columns:

* `kernel`: stable kernel or app case name;
* `dfg_sim_cycles`: DFG-sim optimistic cycle or step estimate;
* `cgra_sim_cycles`: CGRA-sim hardware-aware cycle count.

Optional columns may include workload identity, input identity, mapping
artifact identity, classification, ratio, and diagnostics. Optional
columns must not change the meaning or order of the three required
columns when the compact three-column profile is selected.

If either simulator report is unavailable, the row must contain an
explicit unsupported or diagnostic marker according to the selected
export profile. It must not silently write zero for a missing cycle
value.

If a workload is represented by multiple dataflow graph slices, the
compact table row may aggregate those slices only when all source
reports share the same workload identity and input identity. The DFG
cycle value is the sum of matching DFG-sim reports. The CGRA cycle value
is the sum of matching CGRA-sim reports, and each CGRA-sim report must
reference a valid mapping artifact for its slice.

Identical cycle values across distinct `pass` rows are a correctness
risk, not a benign formatting issue. The comparison or artifact audit
must fail such rows unless they belong to an explicitly documented
equivalence group and the evidence explains why the kernels are
first-principles equivalent for the modeled input, operation family, and
graph shape.

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
* it explains why CGRA-sim differs from the optimistic DFG-sim baseline;
* it can feed PnR and DSE as explicit evidence;
* it can emit the compact cycle table export with `kernel`,
  `dfg_sim_cycles`, and `cgra_sim_cycles` columns.

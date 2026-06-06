# Intermediate Artifact Gates

This document specifies Loom's target intermediate artifact gates. These
artifacts make the compiler, mapping, simulation, RTL, FPA, reporting,
and DSE stack measurable before the final full-stack flow is complete.

## Purpose

Intermediate artifact gates answer this question:

```text
At this point in the tool stack, what concrete summary artifact proves
that the stage produced internally reasonable evidence, and what
diagnostics are emitted when the evidence is absent, inconsistent, or
not yet supported?
```

The artifacts in this spec are portable schema contracts. They do not
define local output paths. Local ignored execution guides may choose
concrete paths for these artifacts.

## Core Rules

Every intermediate artifact gate has:

* artifact kind;
* schema version;
* producer command role;
* required columns or required JSON keys;
* allowed status values;
* missing-value policy;
* unit policy for numeric fields;
* content-audit requirements;
* negative diagnostics.

Artifact existence is not sufficient evidence. A producer may create a
file with invalid, empty, stale, or internally contradictory contents.
Consumers and acceptance gates must inspect the artifact contents.

JSON artifacts are self-describing when they contain a recognized
`kind` field. Canonical filenames remain recommended for portability,
but content audits must be able to classify DFG-sim reports,
CGRA-sim reports, and PnR mapping artifacts from `kind` when local
test or scratch paths use non-canonical filenames.

## Status Values

CSV gates use these baseline status values unless a more specific spec
defines a narrower set:

* `pass`: the row has complete evidence for the requested stage;
* `fail`: the row attempted the stage and failed;
* `unsupported`: the row is outside supported scope and records a
  reason;
* `skipped`: the row was intentionally skipped by policy and records a
  reason;
* `blocked`: the row could not run because an upstream artifact,
  profile, tool, library, or input was missing;
* `not_run`: the row has not been attempted and must not count as pass
  evidence.

`pass` rows must contain real evidence fields. Empty evidence cells in
`pass` rows are invalid.

## Missing-Value Policy

Missing evidence must be explicit. Numeric zero must be used only when
the measured or modeled value is actually zero. Missing cycle, count,
frequency, area, power, energy, route, or mapping evidence must use an
explicit diagnostic or unsupported marker, not numeric zero.

Required identity fields must never be blank in `pass` rows. Optional
identity fields may be blank only when the row status explains why the
optional artifact does not exist.

## CSV Gate Schemas

### Source Compatibility Summary

Purpose: prove ordinary source compatibility before acceleration is
required.

Required first columns:

* `case`;
* `suite`;
* `native_status`;
* `loom_status`;
* `mode`;
* `diagnostic`.

Rules:

* `suite` identifies app, CMSIS-DSP, CMSIS-NN, or another source
  suite.
* `native_status` records ordinary compiler behavior.
* `loom_status` records drop-in Loom driver behavior.
* `mode` distinguishes compatibility, artifact, report-only, and
  acceleration-required modes.
* A `pass` Loom compatibility row requires native compatibility
  evidence for the same case unless the case is Loom-specific.

### Compiler Pipeline Summary

Purpose: prove the source to LLVM IR to raised MLIR to dataflow path.

Required first columns:

* `case`;
* `suite`;
* `llvm_ir_status`;
* `raised_mlir_status`;
* `dataflow_status`;
* `diagnostic`.

Rules:

* A later-stage `pass` requires earlier-stage `pass` for the same row.
* Unsupported lowering must be recorded as `unsupported`, not as a
  successful empty artifact.
* The row should reference artifact identities when the selected export
  profile includes optional columns.

### Dataflow Primitive Coverage

Purpose: prove lowering and DFG-sim coverage for dataflow primitives.

Required first columns:

* `workload`;
* `primitive`;
* `op_count`;
* `dfg_sim_status`;
* `diagnostic`.

Rules:

* `op_count` is a non-negative integer.
* `op_count = 0` is legal only for explicit absence coverage rows; it
  must not be used to claim runtime behavior for that primitive.
* `dfg_sim_status = pass` requires either positive operation coverage
  or an explicit non-runtime coverage classification.

### ADG Hardware Summary

Purpose: prove ADG Builder and Fabric verification for hardware
candidates.

Required first columns:

* `hardware`;
* `topology_class`;
* `node_count`;
* `link_count`;
* `verify_status`;
* `diagnostic`.

Rules:

* `topology_class` distinguishes arbitrary graph, non-mesh, regular
  graph, mesh-like, systolic-like, and custom classes.
* `node_count` and `link_count` are non-negative integers.
* `verify_status = pass` requires positive node count and legal link
  evidence for hardware candidates that are not intentionally empty.
* Coordinates or visualization metadata must not be counted as links.

### PnR Mapping Summary

Purpose: prove PnR emits usable independent mapping artifacts.

Required first columns:

* `workload`;
* `hardware`;
* `mapping_id`;
* `placed_records`;
* `routed_edges`;
* `unrouted_edges`;
* `unplaced_records`;
* `status`.

Rules:

* `placed_records`, `routed_edges`, `unrouted_edges`, and
  `unplaced_records` are non-negative integers.
* `unplaced_records` counts software nodes that could not claim a
  hardware resource. `unrouted_edges` counts software edges whose
  endpoints were placed but whose route could not be established.
* Blocked or unsupported rows must include a diagnostic column explaining
  the missing upstream artifact, profile, or mapping implementation.
* `status = pass` requires a non-empty `mapping_id`, positive placed
  records, no unplaced records, and no unrouted edges unless the mapping
  profile explicitly permits partial mappings.
* A mapping row must agree with the referenced mapping artifact
  identity and verifier result when those optional columns are present.

### Simulator Cycle Summary

Purpose: compare DFG-sim optimistic cycles or steps with CGRA-sim
hardware-aware cycles.

Required first columns:

* `kernel`;
* `dfg_sim_cycles`;
* `cgra_sim_cycles`.

Rules:

* Cycle fields are non-negative numeric values only when the
  corresponding simulator report exists and produced the value.
* Missing simulator evidence is represented by an explicit diagnostic,
  not numeric zero.
* When one kernel or app row is composed from multiple dataflow graph
  slices, the summary value is the sum of the matching DFG-sim report
  cycles for that workload. If CGRA-sim reports are present for those
  slices, the hardware-aware value is the sum of matching CGRA-sim
  report cycles, and each CGRA-sim report must resolve to a pass PnR
  mapping artifact.
* CGRA-sim must not be more optimistic than DFG-sim for comparable
  metrics unless the row records a valid comparability diagnostic.
* DFG-sim cycles must be dynamic-execution metrics, not static graph
  size metrics. For the same workload and dataflow graph, reports with
  larger `dynamic_work_items` must have strictly larger
  `optimistic_cycles` unless the report is explicitly blocked or
  unsupported. This monotonicity rule is independent of any
  cross-kernel equivalence group and cannot be waived by claiming that
  two workloads have similar operation families.
* Distinct `pass` rows with identical DFG-sim or CGRA-sim cycle values
  are invalid by default because they often indicate artifact reuse,
  missing graph coverage, or an over-flat cost model. The only valid
  exception is an explicitly documented equivalence group whose members
  have the same operation family, same modeled input size, same relevant
  graph shape, and matching first-principles audit evidence. For
  example, same-length integer sum-reduction kernels may share a cycle
  value; unrelated kernels such as elementwise arithmetic, mean, and
  norm reductions must not be accepted as equivalent without evidence.

### RTL FPA Summary

Purpose: prove Fabric-to-RTL and FPA evidence are connected.

Required first columns:

* `hardware`;
* `workload`;
* `rtl_lint_status`;
* `rtl_sim_status`;
* `synth_status`;
* `frequency_mhz`;
* `area_um2`;
* `dynamic_power_mw`;
* `leakage_power_mw`.

Rules:

* `frequency_mhz`, `area_um2`, `dynamic_power_mw`, and
  `leakage_power_mw` are numeric only when the corresponding evidence
  exists.
* Missing backend evidence is recorded as unsupported, skipped, or
  blocked according to the selected profile.
* Workload-specific power must identify an activity source in optional
  columns or in the referenced FPA report.

### End-To-End Demonstrator Summary

Purpose: prove the required demonstrator matrix has executable
evidence.

Required first columns:

* `demonstrator`;
* `compat_status`;
* `artifact_status`;
* `mapping_status`;
* `sim_status`;
* `rtl_status`;
* `fpa_status`;
* `report_status`.

Rules:

* Each required demonstrator from
  `docs/spec-end-to-end-demonstrators.md` must have a row.
* Optional-stage gaps must be represented by unsupported or blocked
  statuses, not by successful evidence.
* A full-stack demonstrator row may pass only when its required
  artifact identities are present in the artifact manifest.

### DSE Candidate Summary

Purpose: prove DSE can compare immutable candidates using report
metrics.

Required first columns:

* `candidate`;
* `workload`;
* `hardware`;
* `mapping_id`;
* `objective`;
* `cgra_sim_cycles`;
* `frequency_mhz`;
* `area_um2`;
* `dynamic_power_mw`;
* `energy_nj`;
* `selection_status`.

Rules:

* `selection_status` distinguishes selected, pareto, rejected,
  infeasible, and blocked candidates.
* Derived energy must cite runtime or cycle/frequency and power sources
  in optional columns or referenced reports.
* Candidate rows must refer to immutable candidate artifacts. They must
  not describe mutable search state as final evidence.

### Unsupported Scope Ledger

Purpose: prevent optional-stage gaps from being mistaken for passing
evidence.

Required first columns:

* `stage`;
* `case`;
* `artifact`;
* `reason`;
* `owner`;
* `blocking_input`.

Rules:

* `reason`, `owner`, and `blocking_input` are required when a row is
  used to justify an unsupported, skipped, or blocked stage.
* Unsupported-scope rows must not be counted as ordinary pass evidence.
* A later passing artifact for the same stage and case must supersede
  or close the corresponding unsupported-scope row.

## JSON Gate Schemas

### DFG-Sim Report

Purpose: record pure software dataflow simulation evidence for one
dataflow graph without hardware resource constraints.

Required top-level keys:

* `schema_version`;
* `kind`;
* `workload`;
* `graph`;
* `status`;
* `metric_definition`;
* `operation_semantics_source`;
* `operation_cost_model_source`;
* `optimistic_cycles`;
* `wavefront_steps`;
* `event_count`;
* `dynamic_work_items`;
* `operation_fire_counts`;
* `final_outputs`;
* `diagnostics`.

Rules:

* `kind` must be `dfg_sim_report`.
* `optimistic_cycles` is an optimistic dynamic execution estimate:
  pipeline fill latency plus actual operation fire counts multiplied by
  the SSOT reciprocal-throughput cost model. It must not be a static
  count of graph nodes or scheduled wavefronts.
* `dynamic_work_items` records the modeled dynamic input or token scale
  for the graph execution, such as stream true-emission count or seeded
  token count. It is used by content audit to reject non-physical cycle
  estimates that do not grow with input scale.
* `operation_fire_counts` must be the SSOT accounting source for DFG-sim
  operation counts. CGRA-sim must reuse the same operation semantics and
  cost model, adding hardware constraints instead of redefining the
  primitive operation behavior.
* `wavefront_steps` and `event_count` are supporting diagnostics; they
  must not replace `optimistic_cycles` in simulator cycle summaries.

### PnR Mapping Artifact

Purpose: record one concrete software-to-hardware mapping candidate.

Required top-level keys:

* `schema_version`;
* `kind`;
* `workload`;
* `hardware`;
* `graph`;
* `mapping_id`;
* `status`;
* `placed_records`;
* `routed_edges`;
* `unrouted_edges`;
* `unplaced_records`;
* `config_records`;
* `placements`;
* `routes`;
* `config_bitstream`.

Rules:

* `kind` must be `pnr_mapping`.
* `placed_records`, `routed_edges`, `unrouted_edges`,
  `unplaced_records`, and `config_records` must match the corresponding
  list sizes when those lists are present.
* Each routed edge record must carry stable mapping identity fields,
  producer and consumer binding references, payload kind, and a
  non-empty ordered `segments` list.
* Each route segment must identify its segment id, segment kind, source
  endpoint, and sink endpoint. Optional hardware references may point to
  a Fabric link, module path, resource path, adapter, or buffer.
* Route endpoints and hardware references used as CGRA-sim evidence
  must resolve against the referenced Fabric ADG. String-shaped route
  records are not sufficient when a hardware artifact is available.
* Route configuration records must be keyed by the route `record_id`.
  Producer/consumer software ids alone are not unique enough when two
  SSA values or token edges connect the same producer and consumer
  records.
* A pass mapping artifact with routes lacking segment records is
  invalid, even if `routed_edges` matches the route list size.
* The config bitstream must cover placement configuration and route
  endpoint or segment configuration required by downstream consumers.

### CGRA-Sim Report

Purpose: record hardware-aware simulation evidence for one mapped
workload.

Required top-level keys:

* `schema_version`;
* `kind`;
* `workload`;
* `hardware`;
* `mapping_id`;
* `status`;
* `fidelity_level`;
* `metric_definition`;
* `operation_semantics_source`;
* `dfg_cycles`;
* `route_latency_cycles`;
* `memory_latency_cycles`;
* `temporal_penalty_cycles`;
* `performance_delta_cycles`;
* `hardware_aware_cycles`;
* `cycle_breakdown`;
* `first_principles_checks`.

Rules:

* `kind` must be `cgra_sim_report`.
* `hardware_artifact`, when present, identifies the Fabric ADG input
  artifact used by CGRA-sim and must agree with `hardware`.
* `workload`, `hardware`, and `mapping_id` must resolve to a passing
  PnR mapping artifact for the same candidate. A mapping-summary CSV row
  is not sufficient CGRA-sim provenance because the mapping artifact is
  the SSOT for placements, routes, configuration, and resource sharing.
  A report for one mapping candidate cannot validate cycle evidence for
  a different mapping of the same workload.
* A short hardware symbol is legal only when it resolves to exactly one
  verified hardware artifact. If two Fabric artifacts expose the same
  module symbol, consumers must use an unambiguous artifact-qualified
  hardware reference.
* `route_segments`, when present, counts consumed route segments from
  the mapping artifact. Route latency must be explainable from route
  records and the selected fidelity model.
* `hardware_aware_cycles` must not be smaller than comparable
  `dfg_cycles`.

### Full-Stack Artifact Manifest

Purpose: record artifact identities and fingerprints across the full
traceability path for at least one demonstrator.

Required top-level keys:

* `schema_version`;
* `run_id`;
* `artifacts`;
* `edges`;
* `diagnostics`.

Rules:

* `artifacts` is a list or map of artifact records with kind, id,
  producer, status, and optional fingerprint.
* `edges` records producer-consumer relationships between artifact
  identities.
* Diagnostics must identify missing, stale, unsupported, or
  inconsistent artifacts.

### Artifact Audit Summary

Purpose: record content-reasonableness reviews for intermediate
artifacts.

Required top-level keys:

* `schema_version`;
* `run_id`;
* `artifact_reviews`;
* `cross_artifact_findings`;
* `diagnostics`;
* `verdict`.

Rules:

* `verdict` is `pass` only when every required artifact review passes
  and all cross-artifact findings are resolved.
* `artifact_reviews` names the artifact, schema, rows or entries
  checked, parser checks used, and review finding.
* `cross_artifact_findings` records consistency checks across source,
  mapping, simulator, RTL, FPA, report, and DSE artifacts.
* An audit summary with unresolved contradictions blocks milestone
  acceptance.

## Content Audit Requirements

Every intermediate artifact must pass a content audit before it can be
used as milestone evidence. The audit checks:

* exact header order or JSON keys;
* required row families;
* allowed status values;
* missing-value policy;
* numeric units and reasonable ranges;
* cross-field invariants;
* cross-artifact identity and fingerprint consistency;
* unsupported-scope rows that are counted as passes;
* suspicious all-zero metrics;
* empty pass rows;
* stale run ids;
* impossible simulator, mapping, RTL, FPA, or DSE relationships.

The audit may use automated parsers, deterministic statistics, and
independent read-only review. Sampling alone is insufficient for row
validity. For large artifacts, automated checks must cover every row or
entry even when human-readable review samples representative rows.

## Relationship To Full-Stack Reporting

Intermediate artifacts are evidence inputs for report bundles specified
in `docs/spec-full-stack-reporting.md`. A report bundle may summarize
them, but it does not replace their schemas or content-audit
requirements.

## Acceptance Criteria

The intermediate artifact gate target is complete when:

* every major stack boundary has a named intermediate artifact schema;
* every CSV schema defines required first columns and missing-value
  policy;
* every JSON schema defines required top-level keys and identity rules;
* content audits are required before milestone acceptance;
* missing or unsupported evidence is not encoded as numeric zero or
  pass;
* cross-artifact contradictions block acceptance;
* public specs define portable schemas while local execution guides own
  concrete output paths.

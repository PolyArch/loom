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

JSON artifacts are the preferred program-to-program contract for
non-tabular evidence. CSV and other table exports are projections for
regression tracking, review, or dashboards. A table export that
summarizes JSON evidence must carry enough source identity columns to
recover the source JSON artifact kind, id, schema version, and export
profile.

JSON artifacts are self-describing when they contain a recognized
`kind` field. Canonical filenames remain recommended for portability,
but content audits must be able to classify DFG-sim reports,
CGRA-sim reports, and PnR mapping artifacts from `kind` when local
test or scratch paths use non-canonical filenames.

Configured JSON artifacts must also carry the configuration identity and
configuration fingerprint defined by `docs/spec-config-ssot.md`.
Component-specific configuration-view identities are required when a
producer consumes only a typed subset of the resolved configuration.
CSV gates may project these fields, but CSV projection does not replace
the JSON or MLIR source of truth.

This rule is inherited by every configured JSON schema in this document,
including RTL manifests, EDA reports, normalized FPA reports, full-stack
artifact manifests, and audit summaries, even when a shorter local
required-key list focuses on fields unique to that artifact kind.

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

### ADG Inventory

Purpose: provide the JSON SSOT for generated hardware candidates before
CSV projections, reports, PnR, RTL, FPA, or DSE consume them.

Required top-level JSON fields:

* `schema_version`;
* `kind`;
* `inventory_id`;
* `producer`;
* `candidate_count`;
* `input_artifact_fingerprints`;
* `candidates`;
* `diagnostics`;
* `status`.

Each candidate record must include:

* `candidate_id`;
* `recipe_id`;
* `config_id`;
* `config_fingerprint`;
* `fabric_root`;
* `root_kind`;
* `topology_class`;
* `layout_class`;
* `topology_family`;
* `source_mlir`;
* `source_mlir_fingerprint`;
* `hardware_identity`;
* `construct_coverage`;
* `semantic_connectivity_source`;
* `visual_metadata_role`;
* `coordinates_semantic`;
* `verifier_status`;
* `diagnostic`;
* `downstream_consumers`.

Rules:

* `kind` is `adg_inventory`.
* `candidate_count` matches the number of candidate records.
* `candidate_id` and `hardware_identity` are unique inside the
  inventory.
* `source_mlir` resolves to the emitted Fabric MLIR for the candidate,
  and `source_mlir_fingerprint` must match that file.
* `root_kind = fabric.module` uses `topology_class =
  fabric_module_template` and `semantic_connectivity_source =
  graph_region_ssa`.
* `root_kind = fabric.system` uses `topology_class = fabric_system` and
  `semantic_connectivity_source = fabric.link`.
* `layout_class` classifies candidates as `regular` or `irregular` for
  breadth accounting. It is not a topology shortcut.
* `visual_metadata_role` is `metadata_only` or `absent`.
* `coordinates_semantic` must be `false`. Coordinates, ranks, labels,
  and layout hints are for visualization and human inspection only.
  They must not affect Fabric verification, PnR legality, routing,
  simulation, RTL lowering, FPA, or DSE.
* `construct_coverage` records observed Fabric constructs. FU coverage
  is nested coverage only; FU must not appear as a SpatialCore tile kind.
  The core SpatialCore tile vocabulary remains `pe`, `switch`, and
  `mem`, as owned by the Fabric module specs.
* `downstream_consumers` records each consumer status. A candidate is
  not complete full-stack evidence merely because it appears in the
  inventory; missing consumers must be `blocked`, `unsupported`, or
  otherwise structured non-pass.

### ADG Hardware Summary

Purpose: provide the CSV projection consumed by existing summary,
report, and audit tools for hardware candidates. When an ADG inventory
is available, the hardware summary is a projection of that JSON SSOT.

Required first columns:

* `hardware`;
* `topology_class`;
* `node_count`;
* `link_count`;
* `verify_status`;
* `diagnostic`.

Additional required columns:

* `tile_kinds`;
* `schedule_kinds`;
* `adg_builder_recipe_identity`;
* `node_kinds`.

Rules:

* `topology_class` is `fabric_module_template` for SpatialCore or CGRA
  templates emitted as `fabric.module`, and `fabric_system` for
  system-level hardware candidates emitted as `fabric.system`.
* `node_count` and `link_count` are non-negative integers.
* For `fabric_module_template` rows, `tile_kinds` is a
  semicolon-separated, deterministic set of Fabric SpatialCore tile
  kinds observed in the verified template. The baseline tile kinds are
  `pe`, `switch`, and `mem`.
* For `fabric_module_template` rows, `schedule_kinds` is a
  semicolon-separated, deterministic set of schedule predicates observed
  on those tile kinds. The baseline schedule kinds are `spatial` and
  `temporal`.
* For `fabric_module_template` rows, `node_kinds` must be empty because
  system nodes are not SpatialCore tiles.
* For `fabric_system` rows, `node_kinds` is a semicolon-separated,
  deterministic set of system node kinds observed in the verified
  system. The allowed vocabulary is the verifier-legal system node-kind
  universe owned by `docs/spec-fabric-system-adg.md`; this artifact must
  not hard-code a smaller baseline subset as the target contract.
* For `fabric_system` rows, `tile_kinds` and `schedule_kinds` must be
  empty because SpatialCore tile evidence belongs to `fabric.module`
  rows referenced by the system.
* `adg_builder_recipe_identity` is empty when no ADG Builder recipe is
  known for the candidate. When present, it is a stable identity for the
  recipe that generated the candidate Fabric ADG.
* When projected from ADG inventory, each pass row diagnostic must
  reference the `inventory_id` and `candidate_id` so audits can connect
  the CSV row back to the JSON SSOT.
* `verify_status = pass` requires positive node count. `fabric_system`
  pass rows also require positive `link_count`; `fabric_module_template`
  rows may have `link_count = 0` because module connectivity is
  represented by Graph-region SSA values rather than `fabric.link`
  records.
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
* Summary CSV rows may carry only aggregate unrouted counts; the JSON
  mapping artifact is the SSOT for per-edge unrouted diagnostics.
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

Optional equivalence columns:

* `cycle_equivalence_group`;
* `cycle_equivalence_members`;
* `cycle_equivalence_evidence`.

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
  In the compact CSV projection, that exception is recorded by filling
  `cycle_equivalence_group`, `cycle_equivalence_members`, and
  `cycle_equivalence_evidence` on every duplicated `pass` row. Audit must
  reject duplicate cycle values when those fields are absent, disagree
  across members, or lack matching JSON evidence. The CSV evidence cell is
  explanatory text only: the audit must validate the equivalence facts from
  DFG-sim reports (`dynamic_work_items` and `operation_fire_counts`) and
  CGRA-sim reports (`route_segments` and `memory_latency_cycles`).

### CGRA Status Summary

Purpose: provide row-complete status for app, CMSIS-DSP, CMSIS-NN, and
loombench cases as they move toward CGRA-sim.

Required first columns:

* `suite`;
* `case`;
* `source_row`;
* `software_root`;
* `graph_ids`;
* `dfg_mlir`;
* `dfg_mlir_fingerprint`;
* `required_slice_count`;
* `hardware_system`;
* `spatialcore_template`;
* `mapping_id`;
* `dfg_report`;
* `dfg_report_fingerprint`;
* `dfg_status`;
* `mapping_artifact`;
* `mapping_artifact_fingerprint`;
* `mapping_status`;
* `cgra_report`;
* `cgra_report_fingerprint`;
* `cgra_status`;
* `comparison_report`;
* `comparison_report_fingerprint`;
* `comparison_status`;
* `final_outputs_present`;
* `final_memory_state_present`;
* `status`;
* `diagnostic_class`;
* `owner`;
* `blocking_prerequisite`;
* `diagnostic`.

Rules:

* `suite`, `case`, and `source_row` are the row identity. They must be
  stable across CSV and JSON projections.
* `dfg_mlir` identifies compiler-lowering evidence, such as CMSIS
  lowered MLIR containing `dataflow.graph.func` or
  `dataflow.graph.launch`. It is not a DFG-sim report and must never be
  used to satisfy `dfg_report`.
* For CMSIS rows, `dfg_mlir` is selected by the source row basename,
  matching the drop-in DFG runner's emitted filename. Expected exported
  symbols from the CMSIS target manifest are used to validate MLIR
  identity, not to choose a different evidence filename.
* When a row's diagnostic class says CMSIS DFG MLIR evidence exists,
  `dfg_mlir` and `dfg_mlir_fingerprint` are required and must resolve to
  the referenced MLIR file. This records row-specific compiler evidence
  while keeping simulator stages at `not_run`.
* A CMSIS DFG MLIR row must bind evidence to row identity: the
  referenced MLIR filename basename matches `source_row`, graph ids in
  the row match graph ids discovered from the MLIR, and
  `required_slice_count` matches the graph id count for graph-ready rows.
* If a referenced CMSIS DFG MLIR file does not mention any expected
  symbol for the row, the row is `fail` with an identity-mismatch
  diagnostic. It must not be silently treated as missing evidence for a
  different row.
* A CMSIS row with DFG MLIR graph evidence but without DFG-sim, mapping,
  CGRA-sim, and comparison reports is `blocked`, not `pass`.
* A CMSIS row with DFG MLIR evidence but no dataflow graph launch or
  graph definition is `unsupported` or otherwise structured non-pass.
* `pass` requires DFG-sim report, PnR mapping artifact, CGRA-sim report,
  simulation comparison report, and matching final output or memory-state
  evidence. `dfg_mlir` alone cannot contribute to pass status.
* Default status generation must not silently consume stale local CMSIS
  DFG evidence. A producer that wants to use compiler-lowering evidence
  must provide an explicit evidence directory or an equivalent
  same-run-provenance mechanism.

### RTL FPA Summary

Purpose: prove Fabric-to-RTL and FPA evidence are connected. This CSV
gate is a projection of the RTL manifest, backend reports, activity
records, and normalized FPA JSON report; it is not the FPA source of
truth.

Required first columns:

* `hardware`;
* `workload`;
* `rtl_lint_status`;
* `rtl_sim_status`;
* `synth_status`;
* `frequency_mhz`;
* `area_um2`;
* `dynamic_power_mw`;
* `leakage_power_mw`;
* `fidelity_level`;
* `frequency_source`;
* `area_source`;
* `power_source`;
* `activity_source`.

Rules:

* `frequency_mhz`, `area_um2`, `dynamic_power_mw`, and
  `leakage_power_mw` are numeric only when the corresponding evidence
  exists.
* `fidelity_level` records the FPA evidence level for the numeric
  frequency, area, and power fields.
* `frequency_source`, `area_source`, and `power_source` identify the
  model, report, or backend evidence source used for each metric class.
* Missing backend evidence is recorded as unsupported, skipped, or
  blocked according to the selected profile.
* Workload-specific power must identify an `activity_source` in the
  summary row or in the referenced FPA report.
* The row must identify the normalized FPA JSON report when frequency,
  area, power, or energy values are reported through Loom's FPA
  contract.
* Consumers that need FPA report provenance must follow the normalized
  FPA JSON report identity. The CSV summary is a projection of FPA
  metrics, not the canonical report artifact.

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
* `leakage_power_mw`;
* `energy_nj`;
* `selection_status`.

Rules:

* `selection_status` distinguishes selected, pareto, rejected,
  infeasible, and blocked candidates.
* Derived energy must cite runtime or cycle/frequency and power sources
  in optional columns or referenced reports.
* Candidate rows must refer to immutable candidate artifacts. They must
  not describe mutable search state as final evidence.
* Selected, Pareto, and rejected candidate rows must carry
  `hardware_evidence_kind`. Analytic FPA rows use
  `analytic_model_only`; they may participate in declared low-fidelity
  ranking but must not be counted as backend hardware evidence.
* For a workload made of multiple dataflow graphs, `mapping_id` may
  identify a workload graph-set aggregate mapping artifact. The selected
  row's cycle and energy values must match the aggregate CGRA-sim report
  and simulator cycle summary, while the aggregate artifact preserves
  component graph identities and fingerprints.

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
* `config_id`;
* `config_fingerprint`;
* `component_config_view`;
* `component_config_fingerprint`;
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
* `final_memory_state`;
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
* `config_fingerprint` identifies the canonical resolved configuration
  for the run. `component_config_fingerprint` identifies the DFG-sim
  view containing simulator model parameters, operation semantics, and
  reciprocal-throughput cost model used for the report.
* `wavefront_steps` and `event_count` are supporting diagnostics; they
  must not replace `optimistic_cycles` in simulator cycle summaries.
* A derived workload graph-set DFG report is legal only when
  `aggregation_kind = workload_graph_set`, `graph = workload_graph_set`,
  and the report carries `component_dfg_sim_report_identities` plus
  input fingerprints for those components. Its dynamic counts and cycle
  fields are sums of passing per-graph DFG-sim reports. It must not hide
  or replace the component reports used to derive it.

### PnR Mapping Artifact

Purpose: record one concrete software-to-hardware mapping candidate.

Required top-level keys:

* `schema_version`;
* `kind`;
* `config_id`;
* `config_fingerprint`;
* `component_config_view`;
* `component_config_fingerprint`;
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
* `unrouted_edge_details`;
* `config_bitstream`.

Rules:

* `kind` must be `pnr_mapping`.
* `placed_records`, `routed_edges`, `unrouted_edges`,
  `unplaced_records`, and `config_records` must match the corresponding
  list sizes when those lists are present.
* Each routed edge record must carry stable mapping identity fields,
  producer and consumer binding references, payload kind, and a
  non-empty ordered `segments` list.
* Each unrouted edge record must carry stable mapping identity fields,
  producer and consumer binding references, payload kind, source and
  sink endpoints when they can be resolved, and an actionable diagnostic.
  These records explain `unrouted_edges`; they are blocker evidence, not
  routed-edge evidence.
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
* A pass mapping artifact must have `unrouted_edges = 0` and an empty
  `unrouted_edge_details` list. A failed or blocked artifact must not
  hide unrouted edges behind only a summary count when software and
  hardware endpoints are known.
* The config bitstream must cover placement configuration and route
  endpoint or segment configuration required by downstream consumers.
* `config_fingerprint` identifies the canonical resolved configuration
  for the run. `component_config_fingerprint` identifies the PnR and
  mapping-search configuration view used for the artifact. Downstream
  simulator, report, and DSE consumers must reject mismatched canonical
  configuration fingerprints unless an explicit migration rule proves
  the mismatch irrelevant to the consumed evidence.
* A workload graph-set aggregate mapping artifact is a derived mapping
  candidate for a workload composed of multiple mapped dataflow graphs.
  It must carry `aggregation_kind = workload_graph_set`, a stable
  aggregate `mapping_id`, `component_mapping_ids`,
  `component_mapping_artifact_identities`, and input fingerprints for
  the component mapping artifacts. Placement, route, unrouted,
  unplaced, and config counts must equal the component sums.

### CGRA-Sim Report

Purpose: record hardware-aware simulation evidence for one mapped
workload.

Required top-level keys:

* `schema_version`;
* `kind`;
* `config_id`;
* `config_fingerprint`;
* `component_config_view`;
* `component_config_fingerprint`;
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
* The report must compare its canonical configuration fingerprint
  against consumed DFG-sim and PnR evidence when those fingerprints are
  present. A mismatch is a structured failure or blocked condition, not
  a warning. Component configuration view fingerprints are compared only
  against consumed artifacts that declare a semantically comparable view.
* A derived workload graph-set CGRA-sim report is legal only when
  `aggregation_kind = workload_graph_set`, `mapping_id` names the
  aggregate mapping artifact, and `component_cgra_sim_report_identities`
  plus `component_mapping_ids` identify the passing component reports.
  `dfg_cycles`, `hardware_aware_cycles`, route latency, memory latency,
  temporal penalty, route segment, and config counts must equal the
  component sums. The aggregate report is a consumer-facing workload
  view; the per-graph reports remain the simulator evidence source.

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

### RTL Manifest

Purpose: content-audit the RTL manifest specified by
`docs/spec-rtl-lowering.md` and record the emitted RTL source set and
intended validation mode.

Required top-level keys:

* `schema_version`;
* `kind`;
* `mode`;
* `manifest_id`;
* `source_fabric_adg_identity`;
* `source_hardware_root`;
* `lowering_configuration`;
* `emitted_source_files`;
* `top_level_modules`;
* `generated_packages`;
* `generated_interfaces`;
* `black_box_modules`;
* `behavioral_models`;
* `required_tool_capability_classes`;
* `required_library_profile_classes`;
* `constraints`;
* `activity_hooks`;
* `status`;
* `diagnostics`.

Rules:

* `kind` must be `rtl_manifest`.
* `mode` is either `architecture_rtl` or `mapped_workload_rtl`.
* `architecture_rtl` manifests are generated from Fabric ADG alone.
  Their `mapping_artifact_identity` field is absent unless an export
  profile explicitly records a non-semantic reference.
* `mapped_workload_rtl` manifests consume a mapping artifact for
  configuration, initialization, harness, or workload-bound validation
  artifacts. The mapping artifact must not introduce hardware nodes,
  links, memories, or protocol endpoints.
* `emitted_source_files` entries must use portable relative paths and
  carry fingerprints for the emitted source files.
* A passing architecture RTL manifest must identify a Fabric ADG input,
  at least one emitted SystemVerilog source file, at least one top-level
  module, and the required RTL tool capability classes.
* Behavioral models, black boxes, generated packages, generated
  interfaces, constraints, and activity hooks must be recorded
  explicitly when present.

### EDA Report

Purpose: content-audit normalized EDA tooling output specified by
`docs/spec-eda-tooling.md`. This report records a concrete backend
execution such as `rtl_lint` and gives reporting, FPA, and DSE flows a
stable artifact identity instead of consuming private tool logs.

Required top-level keys:

* `schema_version`;
* `kind`;
* `report_id`;
* `capability_class`;
* `rtl_manifest_identity`;
* `tool_profile_id`;
* `tool_name`;
* `tool_version`;
* `fidelity_level`;
* `command_role`;
* `command_timeout_seconds`;
* `checked_top_modules`;
* `checked_source_files`;
* `input_artifact_fingerprints`;
* `source_file_fingerprints`;
* `returncode`;
* `diagnostic_records`;
* `diagnostics`;
* `status`.

Rules:

* `kind` must be `eda_report`.
* `capability_class` identifies the EDA role from
  `docs/spec-eda-tooling.md`; the first required backend role is
  `rtl_lint`.
* `fidelity_level` records the evidence fidelity provided by the
  backend role. Baseline mappings are owned by
  `docs/spec-eda-tooling.md`; `rtl_lint` is structural RTL evidence and
  `rtl_sim` is functional RTL evidence.
* `command_timeout_seconds` is the positive timeout applied to each
  backend tool invocation.
* `rtl_manifest_identity` must resolve to the RTL manifest consumed by
  the backend run, and the report must fingerprint that manifest.
* Source file fingerprints must match the checked source files recorded
  in the referenced RTL manifest.
* `status = pass` requires an activated backend tool, non-empty
  `tool_version`, zero `returncode`, checked top modules, checked
  source files, and no diagnostic records.
* Missing tools, activation failures, missing sources, backend
  execution failures, and parser failures must be represented as
  structured diagnostic records, not as passing backend evidence.
* Report bundles may consume passing EDA report identities. Blocked,
  unsupported, skipped, or failed EDA reports remain auditable
  artifacts but must not be counted as backend pass evidence.
* EDA lint evidence is not FPA evidence. Analytic FPA estimates remain
  labeled as analytic unless a later normalized FPA report directly
  consumes backend metric evidence according to
  `docs/spec-fpa-estimation.md`.

### Normalized FPA Report

Purpose: content-audit the normalized FPA report specified by
`docs/spec-fpa-estimation.md` and prove it can be consumed by reports
and DSE.

Rules:

* Required top-level keys and metric fields are owned by the Report
  Contract section of `docs/spec-fpa-estimation.md`. This gate must not
  define a second FPA JSON schema.
* `kind` must be `fpa_report`.
* `fidelity_level` follows `docs/spec-fidelity-ladder.md`.
* Backend reports may calibrate an analytic model, but calibrated
  analytic output remains analytic unless the metric is directly
  produced by a backend evidence class.
* Workload and hardware report bundles that cite FPA evidence must
  reference this JSON artifact by identity; they must not use an FPA
  CSV summary identity as the normalized FPA report reference.

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

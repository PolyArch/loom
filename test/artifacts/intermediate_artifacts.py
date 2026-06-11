#!/usr/bin/env python3
"""Intermediate artifact schema writers and deterministic content audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dse_objectives
import runtime_evidence_helpers


BASE_STATUSES = {"pass", "fail", "unsupported", "skipped", "blocked", "not_run"}
SELECTION_STATUSES = {"selected", "pareto", "rejected", "infeasible", "blocked"}
IMPORT_STATES = {"accepted", "deferred", "excluded"}
INVENTORY_STATES = {"ready", "blocked"}
IGNORED_IDENTITIES = {"", "scaffold", "none", None}
DATA_MOVEMENT_POLICIES = {
    "shared_coherent",
    "shared_noncoherent",
    "copy_in_copy_out",
    "device_local",
    "simulated",
    "custom",
}
COHERENCE_REQUIREMENT_BY_POLICY = {
    "shared_coherent": "shared_coherent",
    "shared_noncoherent": "explicit_flush_invalidate",
    "copy_in_copy_out": "copy_boundary",
    "device_local": "device_local",
    "simulated": "simulator_consistent",
    "custom": "custom_policy",
}
SIMULATOR_MEMORY_ADDRESS_SPACE = "simulator::memory_model"
SYNCHRONIZATION_POLICIES = {
    "host_wait",
    "host_fence",
    "device_poll",
}
RUNTIME_INVOCATION_ABI = "loom_runtime_package_v1"
FPA_FIDELITY_LEVELS = {
    "analytic",
    "mapped_activity",
    "rtl_structural",
    "rtl_activity",
    "physical_estimate",
    "fpga_estimate",
    "custom",
    "custom_calibrated",
}
FPA_ACTIVITY_SOURCES = {
    "none",
    "default_toggle",
    "cgra_sim",
    "rtl_waveform",
    "rtl_activity_file",
    "backend_internal",
    "custom",
}
ARTIFACT_EDGE_PAIRS = (
    ("old-app-corpus-inventory", "app-corpus-import-status"),
    ("app-corpus-import-status", "source-compat-summary"),
    ("source-compat-summary", "compiler-pipeline-summary"),
    ("compiler-pipeline-summary", "dataflow-primitive-coverage"),
    ("dataflow-primitive-coverage", "pnr-mapping-summary"),
    ("adg-hardware-summary", "pnr-mapping-summary"),
    ("dataflow-primitive-coverage", "sim-cycle-summary"),
    ("dataflow-primitive-coverage", "rtl-fpa-summary"),
    ("adg-hardware-summary", "rtl-fpa-summary"),
    ("pnr-mapping-summary", "e2e-demonstrator-summary"),
    ("sim-cycle-summary", "e2e-demonstrator-summary"),
    ("rtl-fpa-summary", "e2e-demonstrator-summary"),
    ("pnr-mapping-summary", "dse-candidate-summary"),
    ("sim-cycle-summary", "dse-candidate-summary"),
    ("rtl-fpa-summary", "dse-candidate-summary"),
    ("dataflow-primitive-coverage", "unsupported-scope-ledger"),
    ("pnr-mapping-summary", "unsupported-scope-ledger"),
    ("sim-cycle-summary", "unsupported-scope-ledger"),
    ("rtl-fpa-summary", "unsupported-scope-ledger"),
    ("e2e-demonstrator-summary", "unsupported-scope-ledger"),
    ("dse-candidate-summary", "unsupported-scope-ledger"),
)


@dataclass(frozen=True)
class CsvSchema:
    kind: str
    filename: str
    first_columns: tuple[str, ...]
    status_columns: tuple[str, ...]
    extra_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    identity_columns: tuple[str, ...] = ()
    scaffold_row: tuple[str, ...] = ()


CSV_SCHEMAS: dict[str, CsvSchema] = {
    "old_app_corpus_inventory": CsvSchema(
        kind="old_app_corpus_inventory",
        filename="old-app-corpus-inventory.csv",
        first_columns=(
            "case",
            "main_source",
            "implementation_sources",
            "headers",
            "source_count",
            "header_count",
            "feature_tags",
            "status",
            "diagnostic",
        ),
        status_columns=("status",),
        numeric_columns=("source_count", "header_count"),
        identity_columns=("case",),
        scaffold_row=(
            "scaffold",
            "",
            "",
            "",
            "0",
            "0",
            "",
            "blocked",
            "legacy app corpus inventory is not available yet",
        ),
    ),
    "app_import_status": CsvSchema(
        kind="app_import_status",
        filename="app-corpus-import-status.csv",
        first_columns=("case", "import_state", "manifest_case", "reason", "owner"),
        status_columns=("import_state",),
        identity_columns=("case",),
        scaffold_row=(
            "scaffold",
            "deferred",
            "",
            "legacy app corpus import status is not available yet",
            "test_migration",
        ),
    ),
    "source_compat": CsvSchema(
        kind="source_compat",
        filename="source-compat-summary.csv",
        first_columns=("case", "suite", "native_status", "loom_status", "mode", "diagnostic"),
        status_columns=("native_status", "loom_status"),
        identity_columns=("case", "suite"),
        scaffold_row=(
            "scaffold",
            "app",
            "blocked",
            "blocked",
            "report-only",
            "source compatibility runner scaffold emitted diagnostic-only row",
        ),
    ),
    "compiler_pipeline": CsvSchema(
        kind="compiler_pipeline",
        filename="compiler-pipeline-summary.csv",
        first_columns=(
            "case",
            "suite",
            "llvm_ir_status",
            "raised_mlir_status",
            "dataflow_status",
            "diagnostic",
        ),
        status_columns=("llvm_ir_status", "raised_mlir_status", "dataflow_status"),
        identity_columns=("case", "suite"),
        scaffold_row=(
            "scaffold",
            "app",
            "blocked",
            "blocked",
            "blocked",
            "compiler pipeline artifact runner scaffold emitted diagnostic-only row",
        ),
    ),
    "dataflow_primitive_coverage": CsvSchema(
        kind="dataflow_primitive_coverage",
        filename="dataflow-primitive-coverage.csv",
        first_columns=("workload", "primitive", "op_count", "dfg_sim_status", "diagnostic"),
        status_columns=("dfg_sim_status",),
        numeric_columns=("op_count",),
        identity_columns=("workload", "primitive"),
        scaffold_row=(
            "scaffold",
            "none",
            "0",
            "blocked",
            "DFG-sim primitive coverage scaffold has no executable workload yet",
        ),
    ),
    "adg_hardware": CsvSchema(
        kind="adg_hardware",
        filename="adg-hardware-summary.csv",
        first_columns=("hardware", "topology_class", "node_count", "link_count", "verify_status", "diagnostic"),
        extra_columns=("tile_kinds", "schedule_kinds", "adg_builder_recipe_identity"),
        status_columns=("verify_status",),
        numeric_columns=("node_count", "link_count"),
        identity_columns=("hardware", "topology_class"),
        scaffold_row=(
            "scaffold",
            "arbitrary_graph",
            "0",
            "0",
            "blocked",
            "ADG hardware summary scaffold has no hardware candidate yet",
            "",
            "",
            "",
        ),
    ),
    "pnr_mapping": CsvSchema(
        kind="pnr_mapping",
        filename="pnr-mapping-summary.csv",
        first_columns=(
            "workload",
            "hardware",
            "mapping_id",
            "placed_records",
            "routed_edges",
            "unrouted_edges",
            "unplaced_records",
            "status",
        ),
        status_columns=("status",),
        extra_columns=("diagnostic",),
        numeric_columns=("placed_records", "routed_edges", "unrouted_edges", "unplaced_records"),
        identity_columns=("workload", "hardware", "mapping_id"),
        scaffold_row=(
            "scaffold",
            "scaffold",
            "",
            "",
            "",
            "",
            "",
            "blocked",
            "PnR mapping summary scaffold has no software or hardware candidate yet",
        ),
    ),
    "sim_cycle": CsvSchema(
        kind="sim_cycle",
        filename="sim-cycle-summary.csv",
        first_columns=("kernel", "dfg_sim_cycles", "cgra_sim_cycles"),
        status_columns=("status",),
        extra_columns=("status", "diagnostic"),
        numeric_columns=("dfg_sim_cycles", "cgra_sim_cycles"),
        identity_columns=("kernel",),
        scaffold_row=(
            "scaffold",
            "",
            "",
            "blocked",
            "DFG-sim and CGRA-sim cycle evidence is not available yet",
        ),
    ),
    "rtl_fpa": CsvSchema(
        kind="rtl_fpa",
        filename="rtl-fpa-summary.csv",
        first_columns=(
            "hardware",
            "workload",
            "rtl_lint_status",
            "rtl_sim_status",
            "synth_status",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "fidelity_level",
            "frequency_source",
            "area_source",
            "power_source",
            "activity_source",
        ),
        status_columns=("rtl_lint_status", "rtl_sim_status", "synth_status", "status"),
        extra_columns=("status", "diagnostic"),
        numeric_columns=("frequency_mhz", "area_um2", "dynamic_power_mw", "leakage_power_mw"),
        identity_columns=("hardware", "workload"),
        scaffold_row=(
            "scaffold",
            "scaffold",
            "blocked",
            "blocked",
            "blocked",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "blocked",
            "RTL/FPA backend evidence is not available yet",
        ),
    ),
    "e2e_demonstrator": CsvSchema(
        kind="e2e_demonstrator",
        filename="e2e-demonstrator-summary.csv",
        first_columns=(
            "demonstrator",
            "compat_status",
            "artifact_status",
            "mapping_status",
            "sim_status",
            "rtl_status",
            "fpa_status",
            "report_status",
        ),
        status_columns=(
            "compat_status",
            "artifact_status",
            "mapping_status",
            "sim_status",
            "rtl_status",
            "fpa_status",
            "report_status",
        ),
        extra_columns=("diagnostic",),
        identity_columns=("demonstrator",),
        scaffold_row=(
            "scaffold",
            "blocked",
            "blocked",
            "blocked",
            "blocked",
            "blocked",
            "blocked",
            "blocked",
            "end-to-end demonstrator scaffold has no completed trace yet",
        ),
    ),
    "dse_candidate": CsvSchema(
        kind="dse_candidate",
        filename="dse-candidate-summary.csv",
        first_columns=(
            "candidate",
            "workload",
            "hardware",
            "mapping_id",
            "objective",
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
            "selection_status",
        ),
        status_columns=("selection_status",),
        extra_columns=(
            "unsupported_scope_diagnostics_count",
            "candidate_kind",
            "input_artifacts",
            "input_artifact_fingerprints",
            "output_artifacts",
            "objective_record",
            "metric_records",
            "policy_id",
            "ordering_rule",
            "diagnostic",
        ),
        numeric_columns=(
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
        ),
        identity_columns=("candidate", "workload", "hardware", "mapping_id"),
        scaffold_row=(
            "scaffold",
            "scaffold",
            "scaffold",
            "",
            "none",
            "",
            "",
            "",
            "",
            "",
            "",
            "blocked",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "DSE candidate evidence is not available yet",
        ),
    ),
    "unsupported_scope": CsvSchema(
        kind="unsupported_scope",
        filename="unsupported-scope-ledger.csv",
        first_columns=("stage", "case", "artifact", "reason", "owner", "blocking_input"),
        status_columns=(),
        identity_columns=("stage", "case", "artifact"),
        scaffold_row=(
            "artifact_ladder",
            "scaffold",
            "intermediate_artifact",
            "producer scaffold only",
            "implementation",
            "replace scaffold row with real unsupported-scope evidence",
        ),
    ),
}


JSON_SCHEMAS: dict[str, dict[str, object]] = {
    "artifact_manifest": {
        "filename": "full-stack-artifact-manifest.json",
        "required_keys": {"schema_version", "run_id", "artifacts", "edges", "diagnostics"},
        "scaffold": {
            "schema_version": 1,
            "run_id": "scaffold",
            "artifacts": [],
            "edges": [],
            "diagnostics": [
                {
                    "status": "blocked",
                    "message": "artifact manifest scaffold has no full-stack trace yet",
                }
            ],
        },
    },
    "artifact_audit": {
        "filename": "artifact-audit-summary.json",
        "required_keys": {
            "schema_version",
            "run_id",
            "artifact_reviews",
            "cross_artifact_checks",
            "cross_artifact_findings",
            "diagnostics",
            "verdict",
        },
    },
    "dfg_sim_report": {
        "filename": "dfg-sim-report.json",
        "required_keys": {
            "schema_version",
            "kind",
            "workload",
            "graph",
            "status",
            "metric_definition",
            "operation_semantics_source",
            "operation_cost_model_source",
            "optimistic_cycles",
            "wavefront_steps",
            "event_count",
            "dynamic_work_items",
            "operation_fire_counts",
            "final_outputs",
            "diagnostics",
        },
    },
    "cgra_sim_report": {
        "filename": "cgra-sim-report.json",
        "required_keys": {
            "schema_version",
            "kind",
            "workload",
            "hardware",
            "mapping_id",
            "status",
            "fidelity_level",
            "metric_definition",
            "operation_semantics_source",
            "operation_cost_model_source",
            "difference_classification",
            "hardware_bound_classification",
            "dfg_cycles",
            "modeled_lower_bound_cycles",
            "performance_delta_cycles",
            "route_latency_cycles",
            "memory_latency_cycles",
            "temporal_penalty_cycles",
            "hardware_aware_cycles",
            "cycle_breakdown",
            "unmodeled_constraints",
            "first_principles_checks",
            "diagnostics",
        },
    },
    "sim_comparison_report": {
        "filename": "sim-comparison-report.json",
        "required_keys": {
            "schema_version",
            "kind",
            "comparison_id",
            "workload",
            "runtime_input_identity",
            "dfg_sim_report_identity",
            "cgra_sim_report_identity",
            "mapping_artifact_identity",
            "functional_comparison_status",
            "memory_comparison_status",
            "performance_comparison_status",
            "performance_metric_definitions",
            "difference_classification",
            "explanation_categories",
            "diagnostics",
            "status",
        },
    },
    "rtl_manifest": {
        "filename": "rtl-manifest.json",
        "required_keys": {
            "schema_version",
            "kind",
            "manifest_id",
            "source_fabric_adg_identity",
            "mapping_artifact_identity",
            "lowering_configuration",
            "emitted_source_files",
            "top_level_modules",
            "generated_packages",
            "generated_interfaces",
            "black_box_modules",
            "behavioral_models",
            "required_tool_capability_classes",
            "required_library_profile_classes",
            "constraints",
            "activity_hooks",
            "diagnostics",
            "status",
        },
    },
    "eda_report": {
        "filename": "rtl-eda-report.json",
        "required_keys": {
            "schema_version",
            "kind",
            "report_id",
            "capability_class",
            "rtl_manifest_identity",
            "tool_profile_id",
            "tool_name",
            "tool_version",
            "command_role",
            "checked_top_modules",
            "checked_source_files",
            "input_artifact_fingerprints",
            "source_file_fingerprints",
            "returncode",
            "diagnostic_records",
            "diagnostics",
            "status",
        },
    },
    "pnr_mapping_artifact": {
        "filename": "pnr-mapping.json",
        "required_keys": {
            "schema_version",
            "kind",
            "workload",
            "hardware",
            "graph",
            "mapping_id",
            "status",
            "placed_records",
            "routed_edges",
            "unrouted_edges",
            "unplaced_records",
            "config_records",
            "placements",
            "routes",
            "config_bitstream",
        },
    },
    "runtime_package": {
        "filename": "runtime-package.json",
        "required_keys": {
            "schema_version",
            "kind",
            "package_id",
            "workload",
            "work_package_identity",
            "launch_descriptor_identity",
            "host_program_identity",
            "host_wrapper_identity",
            "host_interface",
            "launch_descriptor",
            "runtime_handle_model",
            "selected_mapping_artifact_identity",
            "fabric_adg_identity",
            "target_profile",
            "runtime_configuration",
            "input_artifact_fingerprints",
            "runtime_report",
            "fallback_policy",
            "fallback_decision",
            "synchronization_mode",
            "data_movement_policy",
            "memory_descriptors",
            "argument_descriptors",
            "required_runtime_features",
            "required_data_movement_policies",
            "required_synchronization_policies",
            "simulator_report_identities",
            "diagnostic_records",
            "diagnostics",
            "status",
        },
    },
    "workload_report_bundle": {
        "filename": "workload-report-bundle.json",
        "required_keys": {
            "schema_version",
            "kind",
            "bundle_id",
            "workload",
            "source_artifact_identity",
            "compiler_command_identity",
            "runtime_input_identity",
            "selected_hardware_candidate_identity",
            "selected_mapping_artifact_identity",
            "runtime_host_interface",
            "runtime_evidence",
            "runtime_fallback_decision",
            "input_artifact_fingerprints",
            "report_status",
            "diagnostic_records",
            "diagnostics",
            "metric_records",
        },
    },
    "hardware_report_bundle": {
        "filename": "hardware-report-bundle.json",
        "required_keys": {
            "schema_version",
            "kind",
            "bundle_id",
            "hardware_candidate_identity",
            "fabric_adg_identity",
            "adg_builder_recipe_identity",
            "rtl_manifest_identity",
            "eda_report_identities",
            "fpa_report_identities",
            "supported_workload_classes",
            "input_artifact_fingerprints",
            "report_status",
            "diagnostic_records",
            "diagnostics",
            "metric_records",
        },
    },
    "dse_report_bundle": {
        "filename": "dse-report-bundle.json",
        "required_keys": {
            "schema_version",
            "kind",
            "dse_run_id",
            "objective_records",
            "candidate_list",
            "selected_candidates",
            "pareto_set",
            "rejected_candidate_summaries",
            "referenced_dse_candidate_artifact_identities",
            "referenced_workload_report_bundle_identities",
            "referenced_hardware_candidate_report_bundle_identities",
            "input_artifact_fingerprints",
            "runtime_evidence_summaries",
            "selected_policy_id",
            "policy_configuration",
            "candidate_ordering_rule",
            "report_status",
            "diagnostic_records",
            "diagnostics",
        },
    },
}


STANDARD_ARTIFACT_PATHS = (
    ("old_app_corpus_inventory", "temp/old-app-corpus-inventory.csv"),
    ("app_import_status", "temp/app-corpus-import-status.csv"),
    ("source_compat", "temp/source-compat-summary.csv"),
    ("compiler_pipeline", "temp/compiler-pipeline-summary.csv"),
    ("dataflow_primitive_coverage", "temp/dataflow-primitive-coverage.csv"),
    ("adg_hardware", "temp/adg-hardware-summary.csv"),
    ("pnr_mapping", "temp/pnr-mapping-summary.csv"),
    ("pnr_mapping_artifact", "temp/pnr-mapping.json"),
    ("dfg_sim_report", "temp/vecsum-dfg-sim-report.json"),
    ("cgra_sim_report", "temp/vecsum-cgra-sim-report.json"),
    ("sim_comparison_report", "temp/sim-comparison-report.json"),
    ("runtime_package", "temp/runtime-package.json"),
    ("sim_cycle", "temp/sim-cycle-summary.csv"),
    ("rtl_manifest", "temp/rtl-manifest.json"),
    ("eda_report", "temp/rtl-eda-report.json"),
    ("rtl_fpa", "temp/rtl-fpa-summary.csv"),
    ("workload_report_bundle", "temp/workload-report-bundle.json"),
    ("hardware_report_bundle", "temp/hardware-report-bundle.json"),
    ("dse_report_bundle", "temp/dse-report-bundle.json"),
    ("e2e_demonstrator", "temp/e2e-demonstrator-summary.csv"),
    ("dse_candidate", "temp/dse-candidate-summary.csv"),
    ("unsupported_scope", "temp/unsupported-scope-ledger.csv"),
)

EMBEDDED_JSON_KIND_ALIASES = {
    "dfg_sim_report": "dfg_sim_report",
    "cgra_sim_report": "cgra_sim_report",
    "sim_comparison_report": "sim_comparison_report",
    "rtl_manifest": "rtl_manifest",
    "eda_report": "eda_report",
    "pnr_mapping": "pnr_mapping_artifact",
    "runtime_package": "runtime_package",
    "workload_report_bundle": "workload_report_bundle",
    "hardware_report_bundle": "hardware_report_bundle",
    "dse_report_bundle": "dse_report_bundle",
    "mapping_set_manifest": "mapping_set_manifest",
}


def schema_for_path(path: Path) -> CsvSchema | None:
    name = path.name
    for schema in CSV_SCHEMAS.values():
        if name == schema.filename or name.endswith("-" + schema.filename):
            return schema
    return None


def artifact_kind_for_path(path: Path) -> str:
    schema = schema_for_path(path)
    if schema is not None:
        return schema.kind
    kind = json_kind_for_path(path)
    if kind is not None:
        return kind
    return "unknown"


def discover_artifact_paths(
    root: Path,
    explicit: Iterable[str],
    *,
    include_unsupported_scope: bool,
) -> list[Path]:
    explicit_paths = [Path(value) for value in explicit]
    if explicit_paths:
        return explicit_paths
    if os.environ.get("LOOM_IGNORE_STANDARD_ARTIFACTS") == "1":
        return []
    return [
        root / relative
        for kind, relative in STANDARD_ARTIFACT_PATHS
        if include_unsupported_scope or kind != "unsupported_scope"
        if (root / relative).is_file()
    ]


def json_kind_for_path(path: Path) -> str | None:
    name = path.name
    for kind, schema in JSON_SCHEMAS.items():
        filename = str(schema["filename"])
        if name == filename or name.endswith("-" + filename):
            return kind
    if not path.is_file() or path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    embedded_kind = data.get("kind")
    if not isinstance(embedded_kind, str):
        return None
    return EMBEDDED_JSON_KIND_ALIASES.get(embedded_kind)


def output_path(raw: str) -> Path:
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def csv_header(kind: str) -> list[str]:
    schema = CSV_SCHEMAS[kind]
    return list(schema.first_columns) + list(schema.extra_columns)


def write_csv_rows(kind: str, output: Path, rows: Iterable[dict[str, str]]) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_header(kind))
        writer.writeheader()
        writer.writerows(rows)


def write_csv(kind: str, output: Path) -> None:
    schema = CSV_SCHEMAS[kind]
    header = csv_header(kind)
    if len(schema.scaffold_row) != len(header):
        raise ValueError(
            f"{kind} scaffold row has {len(schema.scaffold_row)} cells "
            f"but header has {len(header)}"
        )
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(schema.scaffold_row)


def write_json(kind: str, output: Path) -> None:
    data = JSON_SCHEMAS[kind]["scaffold"]
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def row_statuses(schema: CsvSchema, row: dict[str, str]) -> list[tuple[str, str]]:
    statuses: list[tuple[str, str]] = []
    for column in schema.status_columns:
        if column in row and row[column] != "":
            statuses.append((column, row[column]))
    return statuses


def validate_numeric(schema: CsvSchema, row: dict[str, str], diagnostics: list[str], row_index: int) -> None:
    statuses = row_statuses(schema, row)
    has_pass = any(value == "pass" or value in {"selected", "pareto"} for _, value in statuses)
    numeric_values: list[float] = []
    for column in schema.numeric_columns:
        value = row.get(column, "")
        if value == "":
            if has_pass:
                diagnostics.append(f"row {row_index}: pass row has missing numeric evidence in {column}")
            continue
        try:
            numeric = float(value)
        except ValueError:
            diagnostics.append(f"row {row_index}: {column} is not numeric: {value!r}")
            continue
        if numeric < 0:
            diagnostics.append(f"row {row_index}: {column} is negative")
        numeric_values.append(numeric)
    if has_pass and numeric_values and all(value == 0 for value in numeric_values):
        diagnostics.append(f"row {row_index}: pass row has suspicious all-zero numeric evidence")


def validate_statuses(schema: CsvSchema, row: dict[str, str], diagnostics: list[str], row_index: int) -> None:
    for column, value in row_statuses(schema, row):
        if column == "selection_status":
            allowed = SELECTION_STATUSES
        elif column == "import_state":
            allowed = IMPORT_STATES
        elif schema.kind == "old_app_corpus_inventory" and column == "status":
            allowed = INVENTORY_STATES
        else:
            allowed = BASE_STATUSES
        if value not in allowed:
            diagnostics.append(f"row {row_index}: {column} has unknown status {value!r}")


def validate_identity(schema: CsvSchema, row: dict[str, str], diagnostics: list[str], row_index: int) -> None:
    statuses = row_statuses(schema, row)
    has_pass = any(value == "pass" or value in {"selected", "pareto"} for _, value in statuses)
    if not has_pass:
        return
    for column in schema.identity_columns:
        if row.get(column, "") == "":
            diagnostics.append(f"row {row_index}: pass row has blank identity column {column}")


def numeric_value(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def nonnegative_int_cell(row: dict[str, str], column: str) -> int | None:
    value = row.get(column, "")
    if value == "":
        return None
    if not value.isdecimal():
        return None
    return int(value)


def dse_ordering_rule_for_objective(objective: str) -> str:
    return dse_objectives.ordering_rule_for_objective(objective)


def dse_objective_for_known_policy_id(policy_id: str) -> str | None:
    prefix = "deterministic_"
    suffix = "_v1"
    if not policy_id.startswith(prefix) or not policy_id.endswith(suffix):
        return None
    objective = policy_id[len(prefix) : -len(suffix)]
    if dse_objective_semantics(objective) is None:
        return None
    return objective


def dse_objective_semantics(objective: str) -> tuple[str, str] | None:
    return dse_objectives.objective_semantics(objective)


def parse_dse_metric_records(
    metric_records: str,
    diagnostics: list[str],
    row_index: int,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in metric_records.split(";"):
        if entry == "":
            continue
        if "=" not in entry:
            diagnostics.append(f"row {row_index}: metric_records entry {entry!r} has no value")
            continue
        name, value = entry.split("=", 1)
        if name == "" or value == "":
            diagnostics.append(f"row {row_index}: metric_records entry {entry!r} is incomplete")
            continue
        if name in parsed:
            diagnostics.append(f"row {row_index}: metric_records repeats {name}")
        parsed[name] = value
    return parsed


def dse_candidate_metric_id(row: dict[str, str], name: str) -> str | None:
    workload = row.get("workload", "")
    hardware = row.get("hardware", "")
    mapping_id = row.get("mapping_id", "")
    if name in {"cgra_sim_cycles", "energy_nj", "throughput_items_per_s", "performance_per_watt", "performance_per_area"}:
        if not workload:
            return None
        return f"metric::{workload}::{name}"
    if name in {"frequency_mhz", "area_um2", "dynamic_power_mw", "leakage_power_mw"}:
        if not hardware:
            return None
        return f"metric::{hardware}::{name}"
    if name == "unsupported_scope_diagnostics_count":
        if not workload or not hardware or not mapping_id:
            return None
        return f"metric::{workload}::{hardware}::{mapping_id}::{name}"
    return None


def validate_kind_invariants(schema: CsvSchema, row: dict[str, str], diagnostics: list[str], row_index: int) -> None:
    statuses = dict(row_statuses(schema, row))
    if schema.kind == "sim_cycle" and statuses.get("status") == "pass":
        for column in ("dfg_sim_cycles", "cgra_sim_cycles"):
            if row.get(column, "") and nonnegative_int_cell(row, column) is None:
                diagnostics.append(f"row {row_index}: {column} must be a non-negative integer cycle count")
        dfg_cycles = nonnegative_int_cell(row, "dfg_sim_cycles")
        cgra_cycles = nonnegative_int_cell(row, "cgra_sim_cycles")
        if dfg_cycles is None:
            diagnostics.append(f"row {row_index}: sim pass row has no DFG-sim cycles")
        if cgra_cycles is None:
            diagnostics.append(f"row {row_index}: sim pass row has no CGRA-sim cycles")
        if dfg_cycles is not None and cgra_cycles is not None and cgra_cycles < dfg_cycles:
            diagnostics.append(
                f"row {row_index}: CGRA-sim cycles are more optimistic than DFG-sim cycles"
            )
    if schema.kind == "rtl_fpa" and statuses.get("status") == "pass":
        fidelity = row.get("fidelity_level", "")
        if fidelity not in FPA_FIDELITY_LEVELS:
            diagnostics.append(f"row {row_index}: RTL/FPA pass row has unknown fidelity_level")
        for column in ("frequency_source", "area_source", "power_source"):
            if not row.get(column, ""):
                diagnostics.append(f"row {row_index}: RTL/FPA pass row has no {column}")
        activity_source = row.get("activity_source", "")
        if activity_source not in FPA_ACTIVITY_SOURCES:
            diagnostics.append(f"row {row_index}: RTL/FPA pass row has unknown activity_source")
        if (
            row.get("dynamic_power_mw", "") or row.get("leakage_power_mw", "")
        ) and not activity_source:
            diagnostics.append(f"row {row_index}: RTL/FPA power evidence has no activity_source")
    if schema.kind == "dse_candidate" and statuses.get("selection_status") in {"selected", "pareto", "rejected"}:
        required_provenance = (
            "candidate_kind",
            "input_artifacts",
            "input_artifact_fingerprints",
            "output_artifacts",
            "objective_record",
            "metric_records",
            "policy_id",
            "ordering_rule",
        )
        for column in required_provenance:
            if not row.get(column, ""):
                diagnostics.append(f"row {row_index}: DSE candidate row has no {column}")
        candidate_id = row.get("candidate", "")
        mapping_id = row.get("mapping_id", "")
        if candidate_id and mapping_id and not candidate_id.endswith(f"::{mapping_id}"):
            diagnostics.append(f"row {row_index}: DSE candidate id does not include mapping_id")
        objective_record = row.get("objective_record", "")
        if objective_record and not objective_record.startswith("objective::"):
            diagnostics.append(f"row {row_index}: objective_record must use objective:: identity")
        objective = row.get("objective", "")
        if objective and objective_record and objective_record != f"objective::{objective}":
            diagnostics.append(f"row {row_index}: objective_record does not match objective")
        ordering_rule = row.get("ordering_rule", "")
        if objective and ordering_rule and ordering_rule != dse_ordering_rule_for_objective(objective):
            diagnostics.append(f"row {row_index}: ordering_rule does not match objective")
        metric_records = row.get("metric_records", "")
        parsed_metrics = parse_dse_metric_records(metric_records, diagnostics, row_index)
        for metric in (
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
        ):
            if metric_records and metric not in parsed_metrics:
                diagnostics.append(f"row {row_index}: metric_records missing {metric}")
            if metric in parsed_metrics:
                row_value = numeric_value(row, metric)
                metric_value = numeric_value(parsed_metrics, metric)
                if (
                    row_value is None
                    or metric_value is None
                    or abs(row_value - metric_value) > 0.001
                ):
                    diagnostics.append(f"row {row_index}: metric_records {metric} does not match row value")
        diagnostic_count = row.get("unsupported_scope_diagnostics_count", "")
        if objective == "minimize_unsupported_scope_diagnostics":
            if diagnostic_count == "":
                diagnostics.append(
                    f"row {row_index}: unsupported-scope objective needs diagnostic count evidence"
                )
            elif numeric_value(row, "unsupported_scope_diagnostics_count") is None:
                diagnostics.append(
                    f"row {row_index}: unsupported_scope_diagnostics_count is not numeric"
                )
            metric_value = parsed_metrics.get("unsupported_scope_diagnostics_count")
            if metric_value is None:
                diagnostics.append(
                    f"row {row_index}: metric_records missing unsupported_scope_diagnostics_count"
                )
            elif numeric_value(parsed_metrics, "unsupported_scope_diagnostics_count") != numeric_value(
                row,
                "unsupported_scope_diagnostics_count",
            ):
                diagnostics.append(
                    f"row {row_index}: metric_records unsupported_scope_diagnostics_count does not match row value"
                )
    if schema.kind == "pnr_mapping" and statuses.get("status") == "pass":
        placed_records = numeric_value(row, "placed_records")
        unrouted_edges = numeric_value(row, "unrouted_edges")
        unplaced_records = numeric_value(row, "unplaced_records")
        if placed_records is not None and placed_records <= 0:
            diagnostics.append(f"row {row_index}: PnR pass row has no placed records")
        if unrouted_edges is not None and unrouted_edges != 0:
            diagnostics.append(f"row {row_index}: PnR pass row has unrouted edges")
        if unplaced_records is not None and unplaced_records != 0:
            diagnostics.append(f"row {row_index}: PnR pass row has unplaced records")
    if schema.kind == "adg_hardware" and statuses.get("verify_status") == "pass":
        node_count = numeric_value(row, "node_count")
        link_count = numeric_value(row, "link_count")
        topology_class = row.get("topology_class", "")
        tile_kinds = {entry for entry in row.get("tile_kinds", "").split(";") if entry}
        schedule_kinds = {entry for entry in row.get("schedule_kinds", "").split(";") if entry}
        if node_count is not None and node_count <= 0:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has no nodes")
        if not tile_kinds:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has no tile kinds")
        if not tile_kinds <= {"pe", "switch", "mem"}:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has unknown tile kinds")
        if not schedule_kinds:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has no schedule kinds")
        if not schedule_kinds <= {"spatial", "temporal"}:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has unknown schedule kinds")
        if (
            topology_class != "fabric_module_template"
            and link_count is not None
            and link_count <= 0
        ):
            diagnostics.append(f"row {row_index}: ADG hardware pass row has no links")


def validate_sim_cycle_uniqueness(rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    dfg_cycles: dict[int, list[str]] = {}
    cgra_cycles: dict[int, list[str]] = {}
    for row in rows:
        if row.get("status") != "pass":
            continue
        kernel = row.get("kernel", "")
        if not valid_identity(kernel):
            continue
        dfg = nonnegative_int_cell(row, "dfg_sim_cycles")
        cgra = nonnegative_int_cell(row, "cgra_sim_cycles")
        if dfg is not None:
            dfg_cycles.setdefault(dfg, []).append(kernel)
        if cgra is not None:
            cgra_cycles.setdefault(cgra, []).append(kernel)
    validate_unique_sim_cycles("DFG-sim", dfg_cycles, diagnostics)
    validate_unique_sim_cycles("CGRA-sim", cgra_cycles, diagnostics)


def validate_unique_sim_cycles(
    label: str, cycles_by_value: dict[int, list[str]], diagnostics: list[str]
) -> None:
    for cycle, kernels in sorted(cycles_by_value.items(), key=lambda item: item[0]):
        unique_kernels = sorted(set(kernels))
        if len(unique_kernels) > 1:
            diagnostics.append(
                f"{label} cycles {cycle} are shared by multiple kernels "
                f"{unique_kernels}; identical simulator numbers require independent equivalence audit"
            )


def validate_dse_candidate_uniqueness(rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    rows_by_candidate: dict[str, list[int]] = {}
    for index, row in enumerate(rows, start=1):
        candidate = row.get("candidate")
        if not valid_identity(candidate):
            continue
        assert candidate is not None
        rows_by_candidate.setdefault(candidate, []).append(index)
    for candidate, row_indices in sorted(rows_by_candidate.items()):
        if len(row_indices) > 1:
            diagnostics.append(
                f"DSE candidate identity {candidate!r} appears in rows {row_indices}; "
                "candidate records must be unique"
            )


def artifact_reference_candidates(anchor: Path, reference: str) -> list[Path]:
    path = Path(reference)
    if path.is_absolute():
        return [path]
    candidates = [path, anchor.parent / path]
    if path.suffix == "":
        candidates.extend(
            [
                Path(f"{reference}.csv"),
                Path(f"{reference}.json"),
                anchor.parent / f"{reference}.csv",
                anchor.parent / f"{reference}.json",
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        unique.append(candidate)
        seen.add(key)
    return unique


def artifact_reference_exists(anchor: Path, reference: str) -> bool:
    return any(candidate.is_file() for candidate in artifact_reference_candidates(anchor, reference))


def resolve_artifact_reference(anchor: Path, reference: str) -> Path:
    candidates = artifact_reference_candidates(anchor, reference)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    return (anchor.parent / path).resolve()


def artifact_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_id_for_path(path: Path | None) -> str:
    if path is None:
        return ""
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def input_artifact_fingerprints(paths: Iterable[Path | None]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path in paths:
        identity = artifact_id_for_path(path)
        if path is not None and identity and path.is_file():
            fingerprints[identity] = artifact_fingerprint(path)
    return fingerprints


def split_semicolon(value: str) -> list[str]:
    return [entry for entry in value.split(";") if entry]


def parse_dse_input_fingerprints(
    raw: str,
    diagnostics: list[str],
    row_index: int,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if raw == "":
        return parsed
    for entry in raw.split(";"):
        if entry == "":
            diagnostics.append(f"row {row_index}: input_artifact_fingerprints contains an empty entry")
            continue
        if "=" not in entry:
            diagnostics.append(f"row {row_index}: input_artifact_fingerprints entry lacks '='")
            continue
        reference, fingerprint = entry.rsplit("=", 1)
        if reference == "":
            diagnostics.append(f"row {row_index}: input_artifact_fingerprints contains an empty reference")
            continue
        if reference in parsed:
            diagnostics.append(f"row {row_index}: input_artifact_fingerprints repeats {reference!r}")
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(
                f"row {row_index}: input_artifact_fingerprints has invalid fingerprint for {reference!r}"
            )
            continue
        parsed[reference] = fingerprint
    return parsed


def validate_dse_artifact_references(path: Path, rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    current_artifact = path.resolve()
    for index, row in enumerate(rows, start=1):
        if row.get("selection_status") not in {"selected", "pareto", "rejected"}:
            continue
        input_references = split_semicolon(row.get("input_artifacts", ""))
        input_fingerprints = parse_dse_input_fingerprints(
            row.get("input_artifact_fingerprints", ""),
            diagnostics,
            index,
        )
        for column in ("input_artifacts", "output_artifacts"):
            references = row.get(column, "")
            if references == "":
                continue
            for reference in references.split(";"):
                if reference == "":
                    diagnostics.append(f"row {index}: {column} contains an empty reference")
                    continue
                if not artifact_reference_exists(path, reference):
                    diagnostics.append(f"row {index}: {column} reference {reference!r} does not exist")
            if column == "output_artifacts" and references:
                resolved_outputs = {
                    resolve_artifact_reference(path, reference)
                    for reference in references.split(";")
                    if reference
                }
                if current_artifact not in resolved_outputs:
                    diagnostics.append(
                        f"row {index}: output_artifacts does not reference this DSE candidate summary"
                    )
        for reference in input_references:
            if reference not in input_fingerprints:
                diagnostics.append(f"row {index}: input_artifact_fingerprints lacks {reference!r}")
                continue
            resolved = resolve_artifact_reference(path, reference)
            if resolved.is_file():
                actual = artifact_fingerprint(resolved)
                if input_fingerprints[reference] != actual:
                    diagnostics.append(f"row {index}: input_artifact_fingerprints stale for {reference!r}")
        input_reference_set = set(input_references)
        for reference in input_fingerprints:
            if reference not in input_reference_set:
                diagnostics.append(
                    f"row {index}: input_artifact_fingerprints references {reference!r} outside input_artifacts"
                )


def audit_csv(path: Path, schema: CsvSchema) -> dict[str, object]:
    diagnostics: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    if header[: len(schema.first_columns)] != schema.first_columns:
        diagnostics.append(
            f"header first columns {list(header[:len(schema.first_columns)])} "
            f"do not match {list(schema.first_columns)}"
        )
    if schema.kind == "adg_hardware":
        expected_header = tuple(csv_header(schema.kind))
        if header != expected_header:
            diagnostics.append(
                f"header columns {list(header)} do not match {list(expected_header)}"
            )
    if not rows:
        diagnostics.append("artifact has no rows")
    for index, row in enumerate(rows, start=1):
        validate_statuses(schema, row, diagnostics, index)
        validate_numeric(schema, row, diagnostics, index)
        validate_identity(schema, row, diagnostics, index)
        validate_kind_invariants(schema, row, diagnostics, index)
        if any(value == "pass" for _, value in row_statuses(schema, row)):
            diagnostic = row.get("diagnostic", "")
            if diagnostic == "" and schema.kind not in {"sim_cycle", "pnr_mapping"}:
                diagnostics.append(f"row {index}: pass row has no diagnostic or evidence note")
    if schema.kind == "sim_cycle":
        validate_sim_cycle_uniqueness(rows, diagnostics)
    if schema.kind == "dse_candidate":
        validate_dse_candidate_uniqueness(rows, diagnostics)
        validate_dse_artifact_references(path, rows, diagnostics)
    return {
        "artifact": str(path),
        "schema": schema.kind,
        "rows_checked": len(rows),
        "parser_checks": ["csv_header", "status_values", "numeric_policy", "identity_policy"],
        "finding": "pass" if not diagnostics else "fail",
        "diagnostics": diagnostics,
    }


def require_manifest_edge(
    edge_pairs: set[tuple[str, str]],
    diagnostics: list[str],
    left: str,
    right: str,
) -> None:
    if (left, right) not in edge_pairs:
        diagnostics.append(f"artifact manifest missing edge {left} -> {right}")


def manifest_component_for_kind(kind: str) -> str:
    return f"{kind}-producer" if kind else ""


def read_manifest_json_artifact(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def iter_manifest_json_identity_references(
    data: dict[str, object] | None,
    key: str,
    artifact_id_set: set[str],
) -> Iterable[str]:
    if data is None:
        return
    references = data.get(key)
    if not isinstance(references, list):
        return
    for reference in references:
        if isinstance(reference, str) and reference in artifact_id_set:
            yield reference


def iter_manifest_json_string_reference(
    data: dict[str, object] | None,
    key: str,
    artifact_id_set: set[str],
) -> Iterable[str]:
    if data is None:
        return
    reference = data.get(key)
    if isinstance(reference, str) and reference in artifact_id_set:
        yield reference


def iter_manifest_optional_artifact_references(
    data: dict[str, object] | None,
    artifact_id_set: set[str],
) -> Iterable[str]:
    if data is None:
        return
    optional = data.get("optional_artifact_identities")
    if isinstance(optional, dict):
        for reference in optional.values():
            if isinstance(reference, str) and reference in artifact_id_set:
                yield reference
    for key in (
        "source_artifact_identity",
        "compiler_command_identity",
        "selected_mapping_artifact_identity",
    ):
        for reference in iter_manifest_json_string_reference(data, key, artifact_id_set):
            yield reference


def iter_artifact_manifest_required_edges(
    artifact_ids: Iterable[str],
    ids_by_kind: dict[str, list[str]],
    artifact_paths_by_id: dict[str, Path] | None = None,
) -> Iterable[tuple[str, str]]:
    artifact_id_set = set(artifact_ids)
    for left, right in ARTIFACT_EDGE_PAIRS:
        if left in artifact_id_set and right in artifact_id_set:
            yield left, right

    for mapping_id in ids_by_kind.get("pnr_mapping_artifact", []):
        mapping = read_manifest_json_artifact(artifact_paths_by_id.get(mapping_id)) if artifact_paths_by_id else None
        for source_kind in ("dataflow_primitive_coverage", "adg_hardware", "pnr_mapping"):
            for source_id in ids_by_kind.get(source_kind, []):
                yield source_id, mapping_id
        if mapping is not None and is_workload_graph_set_aggregate(mapping):
            for source_id in iter_manifest_json_identity_references(
                mapping,
                "component_mapping_artifact_identities",
                artifact_id_set,
            ):
                yield source_id, mapping_id
        if artifact_paths_by_id is None:
            for cgra_id in ids_by_kind.get("cgra_sim_report", []):
                yield mapping_id, cgra_id
        else:
            for cgra_id in ids_by_kind.get("cgra_sim_report", []):
                cgra = read_manifest_json_artifact(artifact_paths_by_id.get(cgra_id))
                if cgra is None:
                    continue
                if (
                    cgra.get("mapping_id") == mapping.get("mapping_id") if mapping is not None else False
                ):
                    yield mapping_id, cgra_id
        for dse_id in ids_by_kind.get("dse_candidate", []):
            yield mapping_id, dse_id

    for sim_id in ids_by_kind.get("sim_cycle", []):
        for dfg_id in ids_by_kind.get("dfg_sim_report", []):
            yield dfg_id, sim_id
        if sim_id == "sim-cycle-summary":
            for cgra_id in ids_by_kind.get("cgra_sim_report", []):
                yield cgra_id, sim_id

    for dfg_id in ids_by_kind.get("dfg_sim_report", []):
        dfg_report = read_manifest_json_artifact(artifact_paths_by_id.get(dfg_id)) if artifact_paths_by_id else None
        for source_id in ids_by_kind.get("dataflow_primitive_coverage", []):
            yield source_id, dfg_id
        if dfg_report is not None and is_workload_graph_set_aggregate(dfg_report):
            for source_id in iter_manifest_json_identity_references(
                dfg_report,
                "component_dfg_sim_report_identities",
                artifact_id_set,
            ):
                yield source_id, dfg_id

    for cgra_id in ids_by_kind.get("cgra_sim_report", []):
        cgra_report = read_manifest_json_artifact(artifact_paths_by_id.get(cgra_id)) if artifact_paths_by_id else None
        if cgra_report is not None and is_workload_graph_set_aggregate(cgra_report):
            for key in (
                "component_dfg_sim_report_identities",
                "component_cgra_sim_report_identities",
            ):
                for source_id in iter_manifest_json_identity_references(cgra_report, key, artifact_id_set):
                    yield source_id, cgra_id
        for dse_id in ids_by_kind.get("dse_candidate", []):
            yield cgra_id, dse_id

    for rtl_manifest_id in ids_by_kind.get("rtl_manifest", []):
        for hardware_id in ids_by_kind.get("adg_hardware", []):
            yield hardware_id, rtl_manifest_id
        for eda_id in ids_by_kind.get("eda_report", []):
            yield rtl_manifest_id, eda_id
        for rtl_fpa_id in ids_by_kind.get("rtl_fpa", []):
            yield rtl_manifest_id, rtl_fpa_id

    for comparison_id in ids_by_kind.get("sim_comparison_report", []):
        if artifact_paths_by_id is None:
            for source_kind in ("dfg_sim_report", "cgra_sim_report", "pnr_mapping_artifact"):
                for source_id in ids_by_kind.get(source_kind, []):
                    yield source_id, comparison_id
            continue
        comparison = read_manifest_json_artifact(artifact_paths_by_id.get(comparison_id))
        for key in (
            "dfg_sim_report_identity",
            "cgra_sim_report_identity",
            "mapping_artifact_identity",
        ):
            for source_id in iter_manifest_json_string_reference(comparison, key, artifact_id_set):
                yield source_id, comparison_id

    for runtime_id in ids_by_kind.get("runtime_package", []):
        if artifact_paths_by_id is None:
            for source_kind in ("pnr_mapping_artifact", "cgra_sim_report", "sim_comparison_report"):
                for source_id in ids_by_kind.get(source_kind, []):
                    yield source_id, runtime_id
            continue
        runtime = read_manifest_json_artifact(artifact_paths_by_id.get(runtime_id))
        for source_id in iter_manifest_json_string_reference(
            runtime,
            "selected_mapping_artifact_identity",
            artifact_id_set,
        ):
            yield source_id, runtime_id
        for source_id in iter_manifest_json_identity_references(
            runtime,
            "simulator_report_identities",
            artifact_id_set,
        ):
            yield source_id, runtime_id

    for report_id in ids_by_kind.get("workload_report_bundle", []):
        if artifact_paths_by_id is None:
            for source_kind in (
                "source_compat",
                "compiler_pipeline",
                "dataflow_primitive_coverage",
                "adg_hardware",
                "pnr_mapping_artifact",
                "dfg_sim_report",
                "cgra_sim_report",
                "sim_comparison_report",
                "runtime_package",
                "sim_cycle",
                "rtl_manifest",
                "rtl_fpa",
                "dse_candidate",
            ):
                for source_id in ids_by_kind.get(source_kind, []):
                    yield source_id, report_id
        else:
            report = read_manifest_json_artifact(artifact_paths_by_id.get(report_id))
            for source_id in iter_manifest_optional_artifact_references(report, artifact_id_set):
                yield source_id, report_id
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            yield report_id, demonstrator_id

    for hardware_report_id in ids_by_kind.get("hardware_report_bundle", []):
        for source_kind in ("adg_hardware", "rtl_manifest", "rtl_fpa"):
            for source_id in ids_by_kind.get(source_kind, []):
                yield source_id, hardware_report_id
        if artifact_paths_by_id is not None and hardware_report_id in artifact_paths_by_id:
            report = read_manifest_json_artifact(artifact_paths_by_id.get(hardware_report_id))
            for source_id in iter_manifest_json_identity_references(
                report,
                "eda_report_identities",
                artifact_id_set,
            ):
                yield source_id, hardware_report_id
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            yield hardware_report_id, demonstrator_id

    for dse_report_id in ids_by_kind.get("dse_report_bundle", []):
        if artifact_paths_by_id is None or dse_report_id not in artifact_paths_by_id:
            for source_kind in ("dse_candidate", "workload_report_bundle", "hardware_report_bundle"):
                for source_id in ids_by_kind.get(source_kind, []):
                    yield source_id, dse_report_id
            continue
        report = read_manifest_json_artifact(artifact_paths_by_id.get(dse_report_id))
        for key in (
            "referenced_dse_candidate_artifact_identities",
            "referenced_workload_report_bundle_identities",
            "referenced_hardware_candidate_report_bundle_identities",
        ):
            for source_id in iter_manifest_json_identity_references(report, key, artifact_id_set):
                yield source_id, dse_report_id


def manifest_artifact_path(raw_path: object, manifest_path: Path | None) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute() or manifest_path is None:
        return path
    return manifest_path.parent / path


def validate_artifact_manifest_edges(
    data: dict[str, object],
    diagnostics: list[str],
    manifest_path: Path | None = None,
) -> int:
    artifacts = data.get("artifacts")
    edges = data.get("edges")
    if not isinstance(artifacts, list):
        diagnostics.append("artifact manifest artifacts must be a list")
        artifacts = []
    if not isinstance(edges, list):
        diagnostics.append("artifact manifest edges must be a list")
        edges = []

    artifact_ids: set[str] = set()
    artifact_kinds: dict[str, str] = {}
    artifact_fingerprints: dict[str, str] = {}
    artifact_paths_by_id: dict[str, Path] = {}
    ids_by_kind: dict[str, list[str]] = {}
    for index, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, dict):
            diagnostics.append(f"artifact manifest artifact {index} must be an object")
            continue
        identity = artifact.get("id")
        kind = artifact.get("kind")
        if not isinstance(identity, str) or not identity:
            diagnostics.append(f"artifact manifest artifact {index} lacks id")
            continue
        if identity in artifact_ids:
            diagnostics.append(f"artifact manifest duplicate artifact id {identity}")
        artifact_ids.add(identity)
        if not isinstance(kind, str) or not kind:
            diagnostics.append(f"artifact manifest artifact {identity} lacks kind")
            continue
        artifact_kinds[identity] = kind
        fingerprint = artifact.get("fingerprint")
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(f"artifact manifest artifact {identity} lacks valid fingerprint")
        else:
            assert isinstance(fingerprint, str)
            artifact_fingerprints[identity] = fingerprint
        resolved_path = manifest_artifact_path(
            artifact.get("path", artifact.get("logical_path")),
            manifest_path,
        )
        if resolved_path is not None:
            artifact_paths_by_id[identity] = resolved_path
        ids_by_kind.setdefault(kind, []).append(identity)

    edge_pairs: set[tuple[str, str]] = set()
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            diagnostics.append(f"artifact manifest edge {index} must be an object")
            continue
        edge_id = edge.get("id")
        left = edge.get("from")
        right = edge.get("to")
        if not isinstance(edge_id, str) or not edge_id:
            diagnostics.append(f"artifact manifest edge {index} lacks id")
        elif edge_id in edge_ids:
            diagnostics.append(f"artifact manifest duplicate edge id {edge_id}")
        else:
            edge_ids.add(edge_id)
        if not isinstance(left, str) or not left:
            diagnostics.append(f"artifact manifest edge {index} lacks from")
            continue
        if not isinstance(right, str) or not right:
            diagnostics.append(f"artifact manifest edge {index} lacks to")
            continue
        expected_edge_id = f"edge::{left}->{right}"
        if isinstance(edge_id, str) and edge_id and edge_id != expected_edge_id:
            diagnostics.append(f"artifact manifest edge {index} id does not match from/to")
        if left not in artifact_ids:
            diagnostics.append(f"artifact manifest edge {index} has unknown source {left}")
        if right not in artifact_ids:
            diagnostics.append(f"artifact manifest edge {index} has unknown sink {right}")
        producer_kind = edge.get("producer_artifact_kind")
        consumer_kind = edge.get("consumer_artifact_kind")
        if not isinstance(producer_kind, str) or not producer_kind:
            diagnostics.append(f"artifact manifest edge {index} lacks producer_artifact_kind")
        elif left in artifact_kinds and producer_kind != artifact_kinds[left]:
            diagnostics.append(f"artifact manifest edge {index} producer_artifact_kind does not match source")
        if not isinstance(consumer_kind, str) or not consumer_kind:
            diagnostics.append(f"artifact manifest edge {index} lacks consumer_artifact_kind")
        elif right in artifact_kinds and consumer_kind != artifact_kinds[right]:
            diagnostics.append(f"artifact manifest edge {index} consumer_artifact_kind does not match sink")
        producer_component = edge.get("producer_component")
        expected_producer_component = manifest_component_for_kind(str(producer_kind)) if isinstance(producer_kind, str) else ""
        if not isinstance(producer_component, str) or not producer_component:
            diagnostics.append(f"artifact manifest edge {index} lacks producer_component")
        elif producer_component != expected_producer_component:
            diagnostics.append(f"artifact manifest edge {index} producer_component does not match source kind")
        consumer_component = edge.get("consumer_component")
        expected_consumer_component = manifest_component_for_kind(str(consumer_kind)) if isinstance(consumer_kind, str) else ""
        if not isinstance(consumer_component, str) or not consumer_component:
            diagnostics.append(f"artifact manifest edge {index} lacks consumer_component")
        elif consumer_component != expected_consumer_component:
            diagnostics.append(f"artifact manifest edge {index} consumer_component does not match sink kind")
        if edge.get("public_spec_owner") != "docs/spec-full-stack-traceability.md":
            diagnostics.append(f"artifact manifest edge {index} public_spec_owner is invalid")
        if edge.get("schema_or_verifier") != "intermediate_artifact_audit":
            diagnostics.append(f"artifact manifest edge {index} schema_or_verifier is invalid")
        if edge.get("validation_command_role") != "artifact content audit":
            diagnostics.append(f"artifact manifest edge {index} validation_command_role is invalid")
        if edge.get("negative_diagnostic_classes") != ["missing_edge", "stale_fingerprint"]:
            diagnostics.append(f"artifact manifest edge {index} negative_diagnostic_classes is invalid")
        if edge.get("minimal_positive_demonstrator_requirement") != "intermediate artifact chain":
            diagnostics.append(
                f"artifact manifest edge {index} minimal_positive_demonstrator_requirement is invalid"
            )
        input_fingerprints = edge.get("required_input_fingerprints")
        if not isinstance(input_fingerprints, dict):
            diagnostics.append(f"artifact manifest edge {index} required_input_fingerprints must be an object")
        elif input_fingerprints.get(left) != artifact_fingerprints.get(left):
            diagnostics.append(f"artifact manifest edge {index} input fingerprint does not match source")
        output_fingerprints = edge.get("produced_output_fingerprints")
        if not isinstance(output_fingerprints, dict):
            diagnostics.append(f"artifact manifest edge {index} produced_output_fingerprints must be an object")
        elif output_fingerprints.get(right) != artifact_fingerprints.get(right):
            diagnostics.append(f"artifact manifest edge {index} output fingerprint does not match sink")
        edge_pairs.add((left, right))

    required_edge_pairs = set(iter_artifact_manifest_required_edges(artifact_ids, ids_by_kind, artifact_paths_by_id))
    for left, right in sorted(edge_pairs - required_edge_pairs):
        diagnostics.append(f"artifact manifest unexpected edge {left} -> {right}")
    for left, right in required_edge_pairs:
        require_manifest_edge(edge_pairs, diagnostics, left, right)

    return len(artifacts)


def validate_diagnostic_records(
    value: object,
    diagnostics: list[str],
    label: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        diagnostics.append(f"{label} diagnostic_records must be a list")
        return []
    records: list[dict[str, object]] = []
    for index, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            diagnostics.append(f"{label} diagnostic record {index} must be an object")
            continue
        for key in ("diagnostic_id", "diagnostic_class", "component", "severity", "message"):
            if not isinstance(record.get(key), str) or not record.get(key):
                diagnostics.append(f"{label} diagnostic record {index} lacks {key}")
        if record.get("severity") not in {"info", "warning", "error"}:
            diagnostics.append(f"{label} diagnostic record {index} has unknown severity")
        records.append(record)
    return records


RUNTIME_FAILURE_DOMAINS = {
    "compiler_artifacts",
    "runtime_configuration",
    "platform_services",
    "simulator_execution",
    "hardware_execution",
}


def runtime_source_provenance(data: dict[str, object]) -> object:
    host_interface = data.get("host_interface")
    if isinstance(host_interface, dict):
        return host_interface.get("source_provenance")
    return None


def validate_runtime_diagnostic_provenance(
    records: list[dict[str, object]],
    diagnostics: list[str],
    label: str,
    *,
    source_provenance: object,
    host_wrapper_identity: object,
) -> None:
    for index, record in enumerate(records, start=1):
        failure_domain = record.get("failure_domain")
        if not isinstance(failure_domain, str) or failure_domain not in RUNTIME_FAILURE_DOMAINS:
            diagnostics.append(f"{label} diagnostic record {index} has invalid failure_domain")
        if isinstance(source_provenance, str) and source_provenance:
            if record.get("source_provenance") != source_provenance:
                diagnostics.append(f"{label} diagnostic record {index} source_provenance does not match package")
        if isinstance(host_wrapper_identity, str) and host_wrapper_identity:
            if record.get("host_wrapper_identity") != host_wrapper_identity:
                diagnostics.append(f"{label} diagnostic record {index} host_wrapper_identity does not match package")


def validate_runtime_diagnostic_records(
    value: object,
    diagnostics: list[str],
    label: str,
    *,
    source_provenance: object,
    host_wrapper_identity: object,
) -> list[dict[str, object]]:
    records = validate_diagnostic_records(value, diagnostics, label)
    validate_runtime_diagnostic_provenance(
        records,
        diagnostics,
        label,
        source_provenance=source_provenance,
        host_wrapper_identity=host_wrapper_identity,
    )
    return records


def validate_fallback_decision(
    value: object,
    diagnostics: list[str],
    label: str,
    *,
    expected_policy: object | None = None,
    target_profile_id: object | None = None,
    require_complete: bool = False,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} fallback_decision must be an object")
        return
    if not value and not require_complete:
        return
    for key in ("policy", "decision", "target_profile_id", "reason"):
        if not isinstance(value.get(key), str) or not value.get(key):
            diagnostics.append(f"{label} fallback_decision lacks {key}")
    if not isinstance(value.get("fallback_taken"), bool):
        diagnostics.append(f"{label} fallback_decision fallback_taken must be boolean")
    if expected_policy is not None and value.get("policy") != expected_policy:
        diagnostics.append(f"{label} fallback_decision policy does not match fallback_policy")
    if target_profile_id is not None and value.get("target_profile_id") != target_profile_id:
        diagnostics.append(f"{label} fallback_decision target_profile_id does not match target_profile")
    if value.get("decision") not in {"none", "report_only", "host_fallback", "scalar_fallback", "blocked"}:
        diagnostics.append(f"{label} fallback_decision has unknown decision")


def validate_runtime_launch_descriptor(
    value: object,
    data: dict[str, object],
    target_profile: dict[str, object],
    memory_descriptors: list[object],
    argument_descriptors: list[object],
    diagnostics: list[str],
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package launch_descriptor must be an object")
        return
    for key in (
        "descriptor_id",
        "work_package_identity",
        "selected_accelerator_region",
        "logical_thread_domain",
        "selected_mapping_artifact_identity",
        "target_profile_id",
        "fallback_policy",
        "synchronization_mode",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"runtime package launch_descriptor lacks {key}")
    for key in (
        "argument_descriptor_names",
        "memory_descriptor_logical_arguments",
        "scalar_value_descriptors",
    ):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
            diagnostics.append(f"runtime package launch_descriptor {key} must be a string list")
    for key, expected in (
        ("argument_descriptors", argument_descriptors),
        ("memory_descriptors", memory_descriptors),
    ):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            diagnostics.append(f"runtime package launch_descriptor {key} must be an object list")
        elif entries != expected:
            diagnostics.append(f"runtime package launch_descriptor {key} does not match package")
    for key in ("profiling_settings", "trace_settings"):
        settings = value.get(key)
        if not isinstance(settings, dict) or not isinstance(settings.get("enabled"), bool):
            diagnostics.append(f"runtime package launch_descriptor {key} must record enabled boolean")
    expected_pairs = (
        ("descriptor_id", "launch_descriptor_identity"),
        ("work_package_identity", "work_package_identity"),
        ("selected_mapping_artifact_identity", "selected_mapping_artifact_identity"),
        ("fallback_policy", "fallback_policy"),
        ("synchronization_mode", "synchronization_mode"),
    )
    for descriptor_key, package_key in expected_pairs:
        if value.get(descriptor_key) != data.get(package_key):
            diagnostics.append(
                f"runtime package launch_descriptor {descriptor_key} does not match {package_key}"
            )
    if value.get("target_profile_id") != target_profile.get("profile_id"):
        diagnostics.append("runtime package launch_descriptor target_profile_id does not match target_profile")
    argument_names = [
        descriptor.get("name")
        for descriptor in argument_descriptors
        if isinstance(descriptor, dict) and isinstance(descriptor.get("name"), str)
    ]
    if value.get("argument_descriptor_names") != argument_names:
        diagnostics.append("runtime package launch_descriptor argument descriptors do not match package")
    memory_arguments = [
        descriptor.get("logical_argument")
        for descriptor in memory_descriptors
        if isinstance(descriptor, dict) and isinstance(descriptor.get("logical_argument"), str)
    ]
    if value.get("memory_descriptor_logical_arguments") != memory_arguments:
        diagnostics.append("runtime package launch_descriptor memory descriptors do not match package")


def validate_work_package_metadata(
    value: object,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package work_package_metadata must be an object")
        return
    for key in (
        "work_package_identity",
        "workload",
        "selected_accelerator_region",
        "logical_thread_domain",
        "runtime_input_identity",
    ):
        if not isinstance(value.get(key), str) or not value.get(key):
            diagnostics.append(f"runtime package work_package_metadata lacks {key}")
    for key in ("selected_mapping_artifact_identity", "fabric_adg_identity"):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"runtime package work_package_metadata {key} must be a string")
    expected_pairs = (
        ("work_package_identity", data.get("work_package_identity")),
        ("workload", data.get("workload")),
        ("selected_mapping_artifact_identity", data.get("selected_mapping_artifact_identity")),
        ("fabric_adg_identity", data.get("fabric_adg_identity")),
        ("runtime_input_identity", runtime_source_provenance(data)),
    )
    for key, expected in expected_pairs:
        if value.get(key) != expected:
            diagnostics.append(f"runtime package work_package_metadata {key} does not match package")
    launch_descriptor = data.get("launch_descriptor")
    if isinstance(launch_descriptor, dict):
        for key in ("selected_accelerator_region", "logical_thread_domain"):
            if value.get(key) != launch_descriptor.get(key):
                diagnostics.append(f"runtime package work_package_metadata {key} does not match launch_descriptor")


def validate_runtime_handle_model(
    value: object,
    diagnostics: list[str],
    label: str = "runtime package",
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} runtime_handle_model must be an object")
        return
    if value.get("handle_kind") != "host_visible_launch_handle":
        diagnostics.append(f"{label} runtime_handle_model handle_kind must be host_visible_launch_handle")
    if value.get("ir_token_kind") != "not_dataflow_thread_token":
        diagnostics.append(f"{label} runtime_handle_model must not use dataflow thread tokens")
    if not isinstance(value.get("completion_source"), str) or not value.get("completion_source"):
        diagnostics.append(f"{label} runtime_handle_model lacks completion_source")
    operations = value.get("operations")
    if not isinstance(operations, list) or any(not isinstance(operation, str) for operation in operations):
        diagnostics.append(f"{label} runtime_handle_model operations must be a string list")
        return
    for operation in ("query_status", "wait_for_completion", "collect_diagnostics"):
        if operation not in operations:
            diagnostics.append(f"{label} runtime_handle_model lacks {operation}")


def validate_runtime_configuration(
    value: object,
    data: dict[str, object],
    target_profile: dict[str, object],
    diagnostics: list[str],
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package runtime_configuration must be an object")
        return
    for key in (
        "configuration_id",
        "target_profile_id",
        "data_movement_policy",
        "platform_binding_identity",
        "fallback_policy",
        "synchronization_mode",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"runtime package runtime_configuration lacks {key}")
    custom_identity = value.get("custom_data_movement_policy_identity")
    expected_configuration_id = (
        f"runtime-config::{data.get('fallback_policy')}::"
        f"{data.get('data_movement_policy')}::{data.get('synchronization_mode')}"
    )
    if data.get("data_movement_policy") == "custom" and isinstance(custom_identity, str) and custom_identity:
        expected_configuration_id = f"{expected_configuration_id}::{custom_identity}"
    if value.get("configuration_id") != expected_configuration_id:
        diagnostics.append("runtime package runtime_configuration configuration_id does not match policy fields")
    expected_pairs = (
        ("target_profile_id", target_profile.get("profile_id")),
        ("data_movement_policy", data.get("data_movement_policy")),
        ("fallback_policy", data.get("fallback_policy")),
        ("synchronization_mode", data.get("synchronization_mode")),
    )
    for key, expected in expected_pairs:
        if value.get(key) != expected:
            diagnostics.append(f"runtime package runtime_configuration {key} does not match package")
    if data.get("data_movement_policy") == "custom":
        if not isinstance(custom_identity, str) or not custom_identity:
            diagnostics.append(
                "runtime package runtime_configuration lacks custom_data_movement_policy_identity"
            )
    elif custom_identity is not None:
        diagnostics.append(
            "runtime package runtime_configuration custom_data_movement_policy_identity is only valid for custom policy"
        )


def validate_host_interface(
    value: object,
    data: dict[str, object],
    diagnostics: list[str],
    label: str = "runtime package",
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} host_interface must be an object")
        return
    for key in (
        "host_program_identity",
        "host_wrapper_identity",
        "invocation_abi",
        "source_provenance",
    ):
        if not isinstance(value.get(key), str) or not value.get(key):
            diagnostics.append(f"{label} host_interface lacks {key}")
    for key in ("compatibility_mode_requires_runtime", "acceleration_mode_requires_runtime_package"):
        if not isinstance(value.get(key), bool):
            diagnostics.append(f"{label} host_interface {key} must be boolean")
    if value.get("invocation_abi") != RUNTIME_INVOCATION_ABI:
        diagnostics.append(f"{label} host_interface invocation_abi must be {RUNTIME_INVOCATION_ABI}")
    if "host_program_identity" in data and value.get("host_program_identity") != data.get("host_program_identity"):
        diagnostics.append(f"{label} host_interface host_program_identity does not match package")
    if "host_wrapper_identity" in data and value.get("host_wrapper_identity") != data.get("host_wrapper_identity"):
        diagnostics.append(f"{label} host_interface host_wrapper_identity does not match package")
    if value.get("compatibility_mode_requires_runtime") is not False:
        diagnostics.append(f"{label} compatibility mode must not require runtime")
    if value.get("acceleration_mode_requires_runtime_package") is not True:
        diagnostics.append(f"{label} acceleration mode must require runtime package")


def validate_non_executed_runtime_claims(
    value: dict[str, object],
    output_buffer_identities: object,
    diagnostics: list[str],
    label: str,
) -> None:
    if value.get("launch_status") != "not_run" or value.get("target_status") != "not_run":
        diagnostics.append(f"{label} must remain not_run")
    if value.get("runtime_trace_identity") or value.get("profiling_record_identity") or output_buffer_identities:
        diagnostics.append(f"{label} must not claim runtime outputs")


def validate_runtime_report(
    value: object,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package runtime_report must be an object")
        return
    for key in (
        "report_id",
        "host_program_identity",
        "host_wrapper_identity",
        "work_package_identity",
        "launch_descriptor_identity",
        "mapping_artifact_identity",
        "fabric_adg_identity",
        "target_profile_id",
        "memory_policy",
        "synchronization_mode",
        "runtime_trace_identity",
        "profiling_record_identity",
        "launch_status",
        "target_status",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"runtime package runtime_report lacks {key}")
    expected_pairs = (
        ("host_program_identity", "host_program_identity"),
        ("host_wrapper_identity", "host_wrapper_identity"),
        ("work_package_identity", "work_package_identity"),
        ("launch_descriptor_identity", "launch_descriptor_identity"),
        ("mapping_artifact_identity", "selected_mapping_artifact_identity"),
        ("fabric_adg_identity", "fabric_adg_identity"),
        ("memory_policy", "data_movement_policy"),
        ("synchronization_mode", "synchronization_mode"),
    )
    for report_key, package_key in expected_pairs:
        if value.get(report_key) != data.get(package_key):
            diagnostics.append(f"runtime package runtime_report {report_key} does not match package")
    report_custom_policy = value.get("custom_data_movement_policy_identity")
    runtime_configuration = data.get("runtime_configuration")
    runtime_custom_policy = None
    if isinstance(runtime_configuration, dict):
        runtime_custom_policy = runtime_configuration.get("custom_data_movement_policy_identity")
    if data.get("data_movement_policy") == "custom":
        if not isinstance(report_custom_policy, str) or not report_custom_policy:
            diagnostics.append("runtime package runtime_report lacks custom_data_movement_policy_identity")
        elif isinstance(runtime_custom_policy, str) and report_custom_policy != runtime_custom_policy:
            diagnostics.append("runtime package runtime_report custom policy does not match configuration")
    elif report_custom_policy is not None:
        diagnostics.append("runtime package runtime_report custom policy is only valid for custom policy")
    target_profile = data.get("target_profile")
    if isinstance(target_profile, dict) and value.get("target_profile_id") != target_profile.get("profile_id"):
        diagnostics.append("runtime package runtime_report target_profile_id does not match target_profile")
    fallback = value.get("fallback_decision")
    if fallback != data.get("fallback_decision"):
        diagnostics.append("runtime package runtime_report fallback_decision does not match package")
    simulator_reports = value.get("simulator_report_identities")
    if not isinstance(simulator_reports, list) or any(not isinstance(identity, str) for identity in simulator_reports):
        diagnostics.append("runtime package runtime_report simulator_report_identities must be a string list")
    elif simulator_reports != data.get("simulator_report_identities"):
        diagnostics.append("runtime package runtime_report simulator_report_identities does not match package")
    output_buffers = value.get("output_buffer_identities")
    if not isinstance(output_buffers, list) or any(not isinstance(identity, str) for identity in output_buffers):
        diagnostics.append("runtime package runtime_report output_buffer_identities must be a string list")
    validate_runtime_diagnostic_records(
        value.get("diagnostic_records"),
        diagnostics,
        "runtime package runtime_report",
        source_provenance=runtime_source_provenance(data),
        host_wrapper_identity=data.get("host_wrapper_identity"),
    )
    validate_non_executed_runtime_claims(
        value,
        output_buffers,
        diagnostics,
        "runtime package runtime_report",
    )


def validate_report_output_configuration(
    value: object,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package report_output_configuration must be an object")
        return
    runtime_report = data.get("runtime_report")
    if not isinstance(runtime_report, dict):
        runtime_report = {}
    launch_descriptor = data.get("launch_descriptor")
    if not isinstance(launch_descriptor, dict):
        launch_descriptor = {}
    report_identity = value.get("runtime_report_identity")
    if not isinstance(report_identity, str) or not report_identity:
        diagnostics.append("runtime package report_output_configuration lacks runtime_report_identity")
    elif report_identity != runtime_report.get("report_id"):
        diagnostics.append(
            "runtime package report_output_configuration runtime_report_identity does not match runtime_report"
        )
    for key in ("diagnostic_output_enabled", "trace_output_enabled", "profiling_output_enabled"):
        if not isinstance(value.get(key), bool):
            diagnostics.append(f"runtime package report_output_configuration {key} must be boolean")
    if value.get("diagnostic_output_enabled") is not True:
        diagnostics.append("runtime package report_output_configuration must enable diagnostic output")
    trace_settings = launch_descriptor.get("trace_settings")
    trace_enabled = trace_settings.get("enabled") if isinstance(trace_settings, dict) else None
    if value.get("trace_output_enabled") != trace_enabled:
        diagnostics.append("runtime package report_output_configuration trace setting does not match launch_descriptor")
    profiling_settings = launch_descriptor.get("profiling_settings")
    profiling_enabled = profiling_settings.get("enabled") if isinstance(profiling_settings, dict) else None
    if value.get("profiling_output_enabled") != profiling_enabled:
        diagnostics.append(
            "runtime package report_output_configuration profiling setting does not match launch_descriptor"
        )


def validate_runtime_evidence_work_package_metadata(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} work_package_metadata must be an object")
        return
    for key in (
        "work_package_identity",
        "workload",
        "selected_accelerator_region",
        "logical_thread_domain",
        "runtime_input_identity",
    ):
        if not isinstance(value.get(key), str) or not value.get(key):
            diagnostics.append(f"{label} work_package_metadata lacks {key}")
    for key in ("selected_mapping_artifact_identity", "fabric_adg_identity"):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"{label} work_package_metadata {key} must be a string")
    expected_pairs = (
        ("work_package_identity", evidence.get("work_package_identity")),
        ("selected_mapping_artifact_identity", evidence.get("mapping_artifact_identity")),
        ("fabric_adg_identity", evidence.get("fabric_adg_identity")),
    )
    for key, expected in expected_pairs:
        if value.get(key) != expected:
            diagnostics.append(f"{label} work_package_metadata {key} does not match runtime evidence")


def validate_runtime_evidence_host_interface(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    validate_host_interface(
        value,
        {
            "host_program_identity": evidence.get("host_program_identity"),
            "host_wrapper_identity": evidence.get("host_wrapper_identity"),
        },
        diagnostics,
        label,
    )
    if not isinstance(value, dict):
        return
    work_package_metadata = evidence.get("work_package_metadata")
    expected_source = None
    if isinstance(work_package_metadata, dict):
        expected_source = work_package_metadata.get("runtime_input_identity")
    if expected_source is not None and value.get("source_provenance") != expected_source:
        diagnostics.append(f"{label} host_interface source_provenance does not match work_package_metadata")


def validate_runtime_evidence_report_output_configuration(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} report_output_configuration must be an object")
        return
    report_identity = value.get("runtime_report_identity")
    if not isinstance(report_identity, str) or not report_identity:
        diagnostics.append(f"{label} report_output_configuration lacks runtime_report_identity")
    elif report_identity != evidence.get("runtime_report_identity"):
        diagnostics.append(
            f"{label} report_output_configuration runtime_report_identity does not match runtime evidence"
        )
    for key in ("diagnostic_output_enabled", "trace_output_enabled", "profiling_output_enabled"):
        if not isinstance(value.get(key), bool):
            diagnostics.append(f"{label} report_output_configuration {key} must be boolean")
    if value.get("diagnostic_output_enabled") is not True:
        diagnostics.append(f"{label} report_output_configuration must enable diagnostic output")


def validate_runtime_evidence_launch_descriptor(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} launch_descriptor must be an object")
        return
    for key in (
        "descriptor_id",
        "work_package_identity",
        "selected_accelerator_region",
        "logical_thread_domain",
        "selected_mapping_artifact_identity",
        "target_profile_id",
        "fallback_policy",
        "synchronization_mode",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"{label} launch_descriptor lacks {key}")
    for key in (
        "argument_descriptor_names",
        "memory_descriptor_logical_arguments",
        "scalar_value_descriptors",
    ):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
            diagnostics.append(f"{label} launch_descriptor {key} must be a string list")
    for key in ("argument_descriptors", "memory_descriptors"):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            diagnostics.append(f"{label} launch_descriptor {key} must be an object list")
    for key in ("profiling_settings", "trace_settings"):
        settings = value.get(key)
        if not isinstance(settings, dict) or not isinstance(settings.get("enabled"), bool):
            diagnostics.append(f"{label} launch_descriptor {key} must record enabled boolean")
    expected_pairs = (
        ("descriptor_id", evidence.get("launch_descriptor_identity")),
        ("work_package_identity", evidence.get("work_package_identity")),
        ("selected_mapping_artifact_identity", evidence.get("mapping_artifact_identity")),
        ("target_profile_id", evidence.get("target_profile_id")),
        ("fallback_policy", evidence.get("fallback_policy")),
        ("synchronization_mode", evidence.get("synchronization_mode")),
    )
    for descriptor_key, expected in expected_pairs:
        if value.get(descriptor_key) != expected:
            diagnostics.append(f"{label} launch_descriptor {descriptor_key} does not match runtime evidence")
    work_package_metadata = evidence.get("work_package_metadata")
    if isinstance(work_package_metadata, dict):
        for key in ("selected_accelerator_region", "logical_thread_domain"):
            if value.get(key) != work_package_metadata.get(key):
                diagnostics.append(f"{label} launch_descriptor {key} does not match work_package_metadata")
    argument_descriptors = evidence.get("argument_descriptors")
    if isinstance(argument_descriptors, list):
        if value.get("argument_descriptors") != argument_descriptors:
            diagnostics.append(f"{label} launch_descriptor argument_descriptors do not match runtime evidence")
        argument_names = [
            descriptor.get("name")
            for descriptor in argument_descriptors
            if isinstance(descriptor, dict) and isinstance(descriptor.get("name"), str)
        ]
        if value.get("argument_descriptor_names") != argument_names:
            diagnostics.append(f"{label} launch_descriptor argument descriptors do not match runtime evidence")
    memory_descriptors = evidence.get("memory_descriptors")
    if isinstance(memory_descriptors, list):
        if value.get("memory_descriptors") != memory_descriptors:
            diagnostics.append(f"{label} launch_descriptor memory_descriptors do not match runtime evidence")
        memory_arguments = [
            descriptor.get("logical_argument")
            for descriptor in memory_descriptors
            if isinstance(descriptor, dict) and isinstance(descriptor.get("logical_argument"), str)
        ]
        if value.get("memory_descriptor_logical_arguments") != memory_arguments:
            diagnostics.append(f"{label} launch_descriptor memory descriptors do not match runtime evidence")
    report_output = evidence.get("report_output_configuration")
    if isinstance(report_output, dict):
        trace_settings = value.get("trace_settings")
        trace_enabled = trace_settings.get("enabled") if isinstance(trace_settings, dict) else None
        if trace_enabled != report_output.get("trace_output_enabled"):
            diagnostics.append(f"{label} launch_descriptor trace setting does not match report_output_configuration")
        profiling_settings = value.get("profiling_settings")
        profiling_enabled = profiling_settings.get("enabled") if isinstance(profiling_settings, dict) else None
        if profiling_enabled != report_output.get("profiling_output_enabled"):
            diagnostics.append(f"{label} launch_descriptor profiling setting does not match report_output_configuration")


def validate_runtime_evidence_memory_descriptors(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, list):
        diagnostics.append(f"{label} memory_descriptors must be a list")
        return
    data_movement_policy = evidence.get("data_movement_policy")
    runtime_configuration = evidence.get("runtime_configuration")
    runtime_platform_binding = None
    if isinstance(runtime_configuration, dict):
        runtime_platform_binding = runtime_configuration.get("platform_binding_identity")
    for index, descriptor in enumerate(value, start=1):
        if not isinstance(descriptor, dict):
            diagnostics.append(f"{label} memory descriptor {index} must be an object")
            continue
        for key in (
            "logical_argument",
            "host_buffer_identity",
            "direction",
            "policy",
            "runtime_input_identity",
            "element_layout",
            "address_space",
            "coherence_requirement",
            "transfer_policy",
        ):
            if not isinstance(descriptor.get(key), str) or not descriptor.get(key):
                diagnostics.append(f"{label} memory descriptor {index} lacks {key}")
        for key in ("byte_size", "alignment_bytes"):
            if not isinstance(descriptor.get(key), int) or descriptor.get(key) <= 0:
                diagnostics.append(f"{label} memory descriptor {index} has invalid {key}")
        if descriptor.get("policy") != data_movement_policy:
            diagnostics.append(f"{label} memory descriptor {index} policy does not match runtime evidence")
        if descriptor.get("transfer_policy") != data_movement_policy:
            diagnostics.append(f"{label} memory descriptor {index} transfer_policy does not match runtime evidence")
        if descriptor.get("policy") not in DATA_MOVEMENT_POLICIES:
            diagnostics.append(f"{label} memory descriptor {index} has unknown policy")
        expected_coherence = COHERENCE_REQUIREMENT_BY_POLICY.get(data_movement_policy)
        if expected_coherence is not None and descriptor.get("coherence_requirement") != expected_coherence:
            diagnostics.append(f"{label} memory descriptor {index} coherence_requirement does not match policy")
        if (
            data_movement_policy == "simulated"
            and descriptor.get("address_space") != SIMULATOR_MEMORY_ADDRESS_SPACE
        ):
            diagnostics.append(f"{label} memory descriptor {index} address_space does not match simulated policy")
        if isinstance(runtime_platform_binding, str) and runtime_platform_binding:
            if descriptor.get("platform_binding_identity") != runtime_platform_binding:
                diagnostics.append(
                    f"{label} memory descriptor {index} "
                    "platform_binding_identity does not match runtime configuration"
                )
            expected_address_space = f"{runtime_platform_binding}::address_space"
            if descriptor.get("address_space") != expected_address_space:
                diagnostics.append(
                    f"{label} memory descriptor {index} "
                    "address_space does not match platform binding"
                )


def validate_runtime_evidence_argument_descriptors(
    value: object,
    diagnostics: list[str],
    label: str,
    expected_identities_by_name: dict[str, object] | None = None,
) -> None:
    if not isinstance(value, list):
        diagnostics.append(f"{label} argument_descriptors must be a list")
        return
    if expected_identities_by_name is None:
        expected_identities_by_name = {}
    for index, descriptor in enumerate(value, start=1):
        if not isinstance(descriptor, dict):
            diagnostics.append(f"{label} argument descriptor {index} must be an object")
            continue
        for key in ("name", "identity", "descriptor_kind"):
            if not isinstance(descriptor.get(key), str) or not descriptor.get(key):
                diagnostics.append(f"{label} argument descriptor {index} lacks {key}")
        validate_runtime_argument_descriptor_kind(descriptor, diagnostics, label, index)
        validate_runtime_argument_descriptor_identity(
            descriptor,
            diagnostics,
            label,
            index,
            expected_identities_by_name,
        )


RUNTIME_ARTIFACT_DESCRIPTOR_KINDS = {
    "pnr_mapping_artifact",
    "dfg_sim_report",
    "cgra_sim_report",
    "sim_comparison_report",
    "rtl_manifest",
}
EXPECTED_RUNTIME_ARGUMENT_DESCRIPTOR_KIND_BY_NAME = {
    "runtime_input": "test_fixture",
    "mapping_artifact": "pnr_mapping_artifact",
    "dfg_sim_report": "dfg_sim_report",
    "cgra_sim_report": "cgra_sim_report",
    "sim_comparison_report": "sim_comparison_report",
    "rtl_manifest": "rtl_manifest",
}
EXPECTED_REPORT_METRIC_UNIT_BY_CLASS = {
    "workload_size": "items",
    "optimistic_steps": "cycles",
    "hardware_cycles": "cycles",
    "estimated_runtime": "us",
    "throughput": "items_per_s",
    "frequency": "MHz",
    "area": "um2",
    "dynamic_power": "mW",
    "leakage_power": "mW",
    "energy": "nJ",
    "performance_per_watt": "items_per_s_per_w",
    "performance_per_area": "items_per_s_per_um2",
    "hardware_nodes": "count",
    "hardware_links": "count",
}
EXPECTED_WORKLOAD_DERIVATION_BY_METRIC_CLASS = {
    "estimated_runtime": "cycle_frequency_runtime",
    "throughput": "workload_runtime_throughput",
    "energy": "runtime_power_energy",
    "performance_per_watt": "workload_runtime_power_efficiency",
    "performance_per_area": "workload_runtime_area_efficiency",
}


def validate_report_metric_unit(
    metric: dict[str, object],
    diagnostics: list[str],
    label: str,
    index: int,
) -> None:
    metric_class = metric.get("metric_class")
    unit = metric.get("unit")
    if not isinstance(metric_class, str) or not isinstance(unit, str):
        return
    expected = EXPECTED_REPORT_METRIC_UNIT_BY_CLASS.get(metric_class)
    if expected is not None and unit != expected:
        diagnostics.append(f"{label} metric {index} unit does not match metric_class")


def validate_runtime_argument_descriptor_kind(
    descriptor: dict[str, object],
    diagnostics: list[str],
    label: str,
    index: int,
) -> None:
    name = descriptor.get("name")
    expected = EXPECTED_RUNTIME_ARGUMENT_DESCRIPTOR_KIND_BY_NAME.get(name)
    if expected is not None and descriptor.get("descriptor_kind") != expected:
        diagnostics.append(f"{label} argument descriptor {index} descriptor_kind does not match name")


def runtime_argument_identity_expectations(context: dict[str, object]) -> dict[str, object]:
    expectations: dict[str, object] = {}
    work_package_metadata = context.get("work_package_metadata")
    runtime_input_identity = None
    if isinstance(work_package_metadata, dict):
        runtime_input_identity = work_package_metadata.get("runtime_input_identity")
    if not runtime_input_identity:
        runtime_input_identity = runtime_source_provenance(context)
    if isinstance(runtime_input_identity, str) and runtime_input_identity:
        expectations["runtime_input"] = runtime_input_identity
    mapping_identity = context.get("selected_mapping_artifact_identity")
    if not isinstance(mapping_identity, str):
        mapping_identity = context.get("mapping_artifact_identity")
    if isinstance(mapping_identity, str) and mapping_identity:
        expectations["mapping_artifact"] = mapping_identity
    simulator_report_identities = context.get("simulator_report_identities")
    if isinstance(simulator_report_identities, list):
        for identity in simulator_report_identities:
            if not isinstance(identity, str) or not identity:
                continue
            if identity.endswith("dfg-sim-report"):
                expectations["dfg_sim_report"] = identity
            elif identity.endswith("cgra-sim-report"):
                expectations["cgra_sim_report"] = identity
            elif identity.endswith("sim-comparison-report"):
                expectations["sim_comparison_report"] = identity
    return expectations


def validate_runtime_argument_descriptor_identity(
    descriptor: dict[str, object],
    diagnostics: list[str],
    label: str,
    index: int,
    expected_identities_by_name: dict[str, object],
) -> None:
    name = descriptor.get("name")
    expected = expected_identities_by_name.get(name)
    if expected is not None and descriptor.get("identity") != expected:
        diagnostics.append(f"{label} argument descriptor {index} identity does not match name")


def runtime_artifact_input_references(
    mapping_identity: object,
    simulator_report_identities: object,
    argument_descriptors: object,
) -> set[str]:
    references: set[str] = set()
    if isinstance(mapping_identity, str) and mapping_identity:
        references.add(mapping_identity)
    if isinstance(simulator_report_identities, list):
        references.update(
            identity
            for identity in simulator_report_identities
            if isinstance(identity, str) and identity
        )
    if isinstance(argument_descriptors, list):
        for descriptor in argument_descriptors:
            if not isinstance(descriptor, dict):
                continue
            if descriptor.get("descriptor_kind") in RUNTIME_ARTIFACT_DESCRIPTOR_KINDS:
                identity = descriptor.get("identity")
                if isinstance(identity, str) and identity:
                    references.add(identity)
    return references


def runtime_evidence_artifact_input_references(value: dict[str, object]) -> set[str]:
    return runtime_artifact_input_references(
        value.get("mapping_artifact_identity"),
        value.get("simulator_report_identities"),
        value.get("argument_descriptors"),
    )


def validate_runtime_evidence_input_fingerprint_references(
    input_fingerprints: dict[object, object],
    references: set[str],
    diagnostics: list[str],
    label: str,
) -> None:
    for identity in input_fingerprints:
        if identity not in references:
            diagnostics.append(f"{label} input_artifact_fingerprints references {identity!r} outside runtime inputs")
    for reference in references:
        if reference not in input_fingerprints:
            diagnostics.append(f"{label} input_artifact_fingerprints lacks {reference!r}")


def validate_runtime_evidence_required_features(
    value: object,
    diagnostics: list[str],
    label: str,
    require_complete: bool,
) -> None:
    if not isinstance(value, list):
        diagnostics.append(f"{label} required_runtime_features must be a list")
        return
    if any(not isinstance(feature, str) or not feature for feature in value):
        diagnostics.append(f"{label} required_runtime_features entries must be non-empty strings")
    if require_complete and not value:
        diagnostics.append(f"{label} pass needs required_runtime_features")


def validate_runtime_evidence_target_profile(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} target_profile must be an object")
        return
    for key in ("target_kind", "profile_id"):
        if not isinstance(value.get(key), str) or not value.get(key):
            diagnostics.append(f"{label} target_profile lacks {key}")
    if value.get("profile_id") != evidence.get("target_profile_id"):
        diagnostics.append(f"{label} target_profile profile_id does not match runtime evidence")
    if value.get("target_kind") == "simulator":
        simulator = value.get("simulator")
        if not isinstance(simulator, str) or not simulator:
            diagnostics.append(f"{label} simulator target_profile lacks simulator")
        elif simulator not in {"cgra_sim", "dfg_sim", "rtl_sim"}:
            diagnostics.append(f"{label} target_profile has unsupported simulator")
    elif value.get("target_kind") == "hardware":
        if not isinstance(value.get("hardware_backend"), str) or not value.get("hardware_backend"):
            diagnostics.append(f"{label} hardware target_profile lacks hardware_backend")
    elif value.get("target_kind"):
        diagnostics.append(f"{label} target_profile has unsupported target_kind")


def validate_runtime_evidence_configuration(
    value: object,
    evidence: dict[str, object],
    diagnostics: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append(f"{label} runtime_configuration must be an object")
        return
    for key in (
        "configuration_id",
        "target_profile_id",
        "data_movement_policy",
        "platform_binding_identity",
        "fallback_policy",
        "synchronization_mode",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"{label} runtime_configuration lacks {key}")
    custom_identity = value.get("custom_data_movement_policy_identity")
    expected_configuration_id = (
        f"runtime-config::{evidence.get('fallback_policy')}::"
        f"{evidence.get('data_movement_policy')}::{evidence.get('synchronization_mode')}"
    )
    if evidence.get("data_movement_policy") == "custom" and isinstance(custom_identity, str) and custom_identity:
        expected_configuration_id = f"{expected_configuration_id}::{custom_identity}"
    if value.get("configuration_id") != expected_configuration_id:
        diagnostics.append(f"{label} runtime_configuration configuration_id does not match runtime evidence")
    expected_pairs = (
        ("target_profile_id", evidence.get("target_profile_id")),
        ("data_movement_policy", evidence.get("data_movement_policy")),
        ("fallback_policy", evidence.get("fallback_policy")),
        ("synchronization_mode", evidence.get("synchronization_mode")),
    )
    for key, expected in expected_pairs:
        if value.get(key) != expected:
            diagnostics.append(f"{label} runtime_configuration {key} does not match runtime evidence")
    if evidence.get("data_movement_policy") == "custom":
        if not isinstance(custom_identity, str) or not custom_identity:
            diagnostics.append(f"{label} runtime_configuration lacks custom_data_movement_policy_identity")
    elif custom_identity is not None:
        diagnostics.append(
            f"{label} runtime_configuration custom_data_movement_policy_identity is only valid for custom policy"
        )


def validate_runtime_evidence(
    path: Path,
    value: object,
    diagnostics: list[str],
    require_complete: bool,
) -> None:
    if not isinstance(value, dict):
        diagnostics.append("workload report bundle runtime_evidence must be an object")
        return
    required_keys = (
        "runtime_package_identity",
        "runtime_report_identity",
        "host_program_identity",
        "host_wrapper_identity",
        "host_interface",
        "runtime_handle_model",
        "work_package_metadata",
        "work_package_identity",
        "launch_descriptor_identity",
        "launch_descriptor",
        "mapping_artifact_identity",
        "fabric_adg_identity",
        "target_profile_id",
        "target_profile",
        "fallback_policy",
        "launch_status",
        "target_status",
        "runtime_trace_identity",
        "profiling_record_identity",
        "data_movement_policy",
        "synchronization_mode",
        "memory_descriptors",
        "argument_descriptors",
        "runtime_configuration",
        "required_runtime_features",
        "output_buffer_identities",
        "simulator_report_identities",
        "diagnostic_records",
        "report_output_configuration",
        "input_artifact_fingerprints",
        "required_data_movement_policies",
        "required_synchronization_policies",
        "fallback_decision",
    )
    for key in required_keys:
        if key not in value:
            diagnostics.append(f"workload report bundle runtime_evidence lacks {key}")
    for key in (
        "runtime_package_identity",
        "runtime_report_identity",
        "launch_status",
        "target_status",
        "runtime_trace_identity",
        "profiling_record_identity",
        "data_movement_policy",
        "synchronization_mode",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"workload report bundle runtime_evidence {key} must be a string")
    for key in (
        "host_program_identity",
        "host_wrapper_identity",
        "work_package_identity",
        "launch_descriptor_identity",
        "mapping_artifact_identity",
        "fabric_adg_identity",
        "target_profile_id",
        "fallback_policy",
    ):
        if not isinstance(value.get(key), str):
            diagnostics.append(f"workload report bundle runtime_evidence {key} must be a string")
    validate_runtime_handle_model(
        value.get("runtime_handle_model"),
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_work_package_metadata(
        value.get("work_package_metadata"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_host_interface(
        value.get("host_interface"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_report_output_configuration(
        value.get("report_output_configuration"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_launch_descriptor(
        value.get("launch_descriptor"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_memory_descriptors(
        value.get("memory_descriptors"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_argument_descriptors(
        value.get("argument_descriptors"),
        diagnostics,
        "workload report bundle runtime_evidence",
        runtime_argument_identity_expectations(value),
    )
    validate_runtime_evidence_target_profile(
        value.get("target_profile"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_configuration(
        value.get("runtime_configuration"),
        value,
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    validate_runtime_evidence_required_features(
        value.get("required_runtime_features"),
        diagnostics,
        "workload report bundle runtime_evidence",
        require_complete,
    )
    data_movement_policy = value.get("data_movement_policy")
    if data_movement_policy not in DATA_MOVEMENT_POLICIES:
        diagnostics.append("workload report bundle runtime_evidence has unknown data_movement_policy")
    synchronization_mode = value.get("synchronization_mode")
    if synchronization_mode not in SYNCHRONIZATION_POLICIES:
        diagnostics.append("workload report bundle runtime_evidence has unknown synchronization_mode")
    custom_identity = value.get("custom_data_movement_policy_identity")
    if data_movement_policy == "custom":
        if not isinstance(custom_identity, str) or not custom_identity:
            diagnostics.append("workload report bundle runtime_evidence lacks custom_data_movement_policy_identity")
    elif custom_identity is not None:
        diagnostics.append(
            "workload report bundle runtime_evidence custom_data_movement_policy_identity is only valid for custom policy"
        )
    outputs = value.get("output_buffer_identities")
    if not isinstance(outputs, list) or any(not isinstance(identity, str) for identity in outputs):
        diagnostics.append("workload report bundle runtime_evidence output_buffer_identities must be a string list")
    simulator_reports = value.get("simulator_report_identities")
    if not isinstance(simulator_reports, list) or any(not isinstance(identity, str) for identity in simulator_reports):
        diagnostics.append("workload report bundle runtime_evidence simulator_report_identities must be a string list")
    validate_diagnostic_records(
        value.get("diagnostic_records"),
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    required_data_movement_policies = value.get("required_data_movement_policies")
    if not isinstance(required_data_movement_policies, list):
        diagnostics.append("workload report bundle runtime_evidence required_data_movement_policies must be a list")
        required_data_movement_policies = []
    elif any(
        not isinstance(policy, str) or not policy
        for policy in required_data_movement_policies
    ):
        diagnostics.append(
            "workload report bundle runtime_evidence required_data_movement_policies entries must be non-empty strings"
        )
    else:
        for policy in required_data_movement_policies:
            if policy not in DATA_MOVEMENT_POLICIES:
                diagnostics.append(
                    f"workload report bundle runtime_evidence required_data_movement_policies has unknown policy {policy}"
                )
    required_synchronization_policies = value.get("required_synchronization_policies")
    if not isinstance(required_synchronization_policies, list):
        diagnostics.append("workload report bundle runtime_evidence required_synchronization_policies must be a list")
        required_synchronization_policies = []
    elif any(
        not isinstance(policy, str) or not policy
        for policy in required_synchronization_policies
    ):
        diagnostics.append(
            "workload report bundle runtime_evidence required_synchronization_policies entries must be non-empty strings"
        )
    else:
        for policy in required_synchronization_policies:
            if policy not in SYNCHRONIZATION_POLICIES:
                diagnostics.append(
                    "workload report bundle runtime_evidence "
                    f"required_synchronization_policies has unknown policy {policy}"
                )
    if data_movement_policy not in required_data_movement_policies:
        diagnostics.append(
            "workload report bundle runtime_evidence required_data_movement_policies omits data_movement_policy"
        )
    if synchronization_mode not in required_synchronization_policies:
        diagnostics.append(
            "workload report bundle runtime_evidence required_synchronization_policies omits synchronization_mode"
        )
    input_fingerprints = value.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append("workload report bundle runtime_evidence input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    else:
        for identity, fingerprint in input_fingerprints.items():
            if not isinstance(identity, str) or not identity:
                diagnostics.append("workload report bundle runtime_evidence input_artifact_fingerprints has invalid identity")
                continue
            if not valid_sha256_hex(fingerprint):
                diagnostics.append(
                    "workload report bundle runtime_evidence "
                    f"input_artifact_fingerprints has invalid fingerprint for {identity}"
                )
                continue
            resolved = resolve_artifact_identity_reference(path, identity)
            if resolved is not None and fingerprint != artifact_fingerprint(resolved):
                diagnostics.append(
                    "workload report bundle runtime_evidence "
                    f"input_artifact_fingerprints stale for {identity!r}"
                )
    validate_runtime_evidence_input_fingerprint_references(
        input_fingerprints,
        runtime_evidence_artifact_input_references(value),
        diagnostics,
        "workload report bundle runtime_evidence",
    )
    fallback = value.get("fallback_decision")
    if not isinstance(fallback, dict):
        diagnostics.append("workload report bundle runtime_evidence fallback_decision must be an object")
        fallback = {}
    validate_fallback_decision(
        fallback,
        diagnostics,
        "workload report bundle runtime_evidence",
        expected_policy=value.get("fallback_policy"),
        target_profile_id=value.get("target_profile_id"),
        require_complete=require_complete,
    )
    if require_complete and not value.get("runtime_report_identity"):
        diagnostics.append("workload report bundle pass needs runtime_report_identity")
    if require_complete and not input_fingerprints:
        diagnostics.append("workload report bundle pass needs runtime input_artifact_fingerprints")
    if fallback.get("decision") == "report_only":
        validate_non_executed_runtime_claims(
            value,
            outputs,
            diagnostics,
            "workload report bundle report_only runtime evidence",
        )


def referenced_workload_runtime_evidence(path: Path, identity: str) -> dict[str, object] | None:
    resolved = resolve_artifact_identity_reference(path, identity)
    if resolved is None:
        return None
    try:
        report = json.loads(resolved.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(report, dict) or report.get("kind") != "workload_report_bundle":
        return None
    evidence = report.get("runtime_evidence")
    return evidence if isinstance(evidence, dict) else None


def validate_runtime_evidence_summaries(
    path: Path,
    value: object,
    diagnostics: list[str],
    require_complete: bool,
    referenced_workload_report_identities: set[str] | None = None,
) -> None:
    if not isinstance(value, list):
        diagnostics.append("DSE report bundle runtime_evidence_summaries must be a list")
        return
    if require_complete and not value:
        diagnostics.append("DSE report bundle pass needs runtime_evidence_summaries")
    for index, summary in enumerate(value, start=1):
        if not isinstance(summary, dict):
            diagnostics.append(f"DSE report bundle runtime evidence summary {index} must be an object")
            continue
        workload_report_identity = summary.get("workload_report_bundle_identity")
        if (
            referenced_workload_report_identities is not None
            and isinstance(workload_report_identity, str)
            and workload_report_identity
            and workload_report_identity not in referenced_workload_report_identities
        ):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "workload_report_bundle_identity is not a referenced workload report"
            )
        if isinstance(workload_report_identity, str) and workload_report_identity:
            expected_evidence = referenced_workload_runtime_evidence(path, workload_report_identity)
            if expected_evidence is not None:
                for key, expected in expected_evidence.items():
                    if summary.get(key) != expected:
                        diagnostics.append(
                            f"DSE report bundle runtime evidence summary {index} {key} "
                            "does not match referenced workload report"
                        )
        for key in (
            "workload_report_bundle_identity",
            "runtime_package_identity",
            "runtime_report_identity",
            "launch_status",
            "target_status",
            "data_movement_policy",
            "synchronization_mode",
            "runtime_handle_model",
            "work_package_metadata",
            "host_interface",
            "launch_descriptor",
            "report_output_configuration",
            "memory_descriptors",
            "argument_descriptors",
            "target_profile",
            "runtime_configuration",
            "required_runtime_features",
        ):
            if key == "runtime_handle_model":
                validate_runtime_handle_model(
                    summary.get(key),
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "work_package_metadata":
                validate_runtime_evidence_work_package_metadata(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "host_interface":
                validate_runtime_evidence_host_interface(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "launch_descriptor":
                validate_runtime_evidence_launch_descriptor(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "report_output_configuration":
                validate_runtime_evidence_report_output_configuration(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "memory_descriptors":
                validate_runtime_evidence_memory_descriptors(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "argument_descriptors":
                validate_runtime_evidence_argument_descriptors(
                    summary.get(key),
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                    runtime_argument_identity_expectations(summary),
                )
            elif key == "target_profile":
                validate_runtime_evidence_target_profile(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "runtime_configuration":
                validate_runtime_evidence_configuration(
                    summary.get(key),
                    summary,
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                )
            elif key == "required_runtime_features":
                validate_runtime_evidence_required_features(
                    summary.get(key),
                    diagnostics,
                    f"DSE report bundle runtime evidence summary {index}",
                    require_complete,
                )
            elif not isinstance(summary.get(key), str) or not summary.get(key):
                diagnostics.append(f"DSE report bundle runtime evidence summary {index} lacks {key}")
        for key in (
            "host_program_identity",
            "host_wrapper_identity",
            "work_package_identity",
            "launch_descriptor_identity",
            "mapping_artifact_identity",
            "fabric_adg_identity",
            "target_profile_id",
            "fallback_policy",
        ):
            if not isinstance(summary.get(key), str):
                diagnostics.append(f"DSE report bundle runtime evidence summary {index} lacks {key}")
        for key in ("runtime_trace_identity", "profiling_record_identity"):
            if not isinstance(summary.get(key), str):
                diagnostics.append(f"DSE report bundle runtime evidence summary {index} lacks {key}")
        data_movement_policy = summary.get("data_movement_policy")
        if data_movement_policy not in DATA_MOVEMENT_POLICIES:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} has unknown data_movement_policy"
            )
        synchronization_mode = summary.get("synchronization_mode")
        if synchronization_mode not in SYNCHRONIZATION_POLICIES:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} has unknown synchronization_mode"
            )
        custom_identity = summary.get("custom_data_movement_policy_identity")
        if data_movement_policy == "custom":
            if not isinstance(custom_identity, str) or not custom_identity:
                diagnostics.append(
                    f"DSE report bundle runtime evidence summary {index} "
                    "lacks custom_data_movement_policy_identity"
                )
        elif custom_identity is not None:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "custom_data_movement_policy_identity is only valid for custom policy"
            )
        required_data_movement_policies = summary.get("required_data_movement_policies")
        if not isinstance(required_data_movement_policies, list):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} required_data_movement_policies must be a list"
            )
            required_data_movement_policies = []
        elif any(
            not isinstance(policy, str) or not policy
            for policy in required_data_movement_policies
        ):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_data_movement_policies entries must be non-empty strings"
            )
        else:
            for policy in required_data_movement_policies:
                if policy not in DATA_MOVEMENT_POLICIES:
                    diagnostics.append(
                        f"DSE report bundle runtime evidence summary {index} "
                        f"required_data_movement_policies has unknown policy {policy}"
                    )
        required_synchronization_policies = summary.get("required_synchronization_policies")
        if not isinstance(required_synchronization_policies, list):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} required_synchronization_policies must be a list"
            )
            required_synchronization_policies = []
        elif any(
            not isinstance(policy, str) or not policy
            for policy in required_synchronization_policies
        ):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_synchronization_policies entries must be non-empty strings"
            )
        else:
            for policy in required_synchronization_policies:
                if policy not in SYNCHRONIZATION_POLICIES:
                    diagnostics.append(
                        f"DSE report bundle runtime evidence summary {index} "
                        f"required_synchronization_policies has unknown policy {policy}"
                    )
        if data_movement_policy not in required_data_movement_policies:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_data_movement_policies omits data_movement_policy"
            )
        if synchronization_mode not in required_synchronization_policies:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_synchronization_policies omits synchronization_mode"
            )
        outputs = summary.get("output_buffer_identities")
        if not isinstance(outputs, list) or any(not isinstance(identity, str) for identity in outputs):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} output_buffer_identities must be a string list"
            )
        simulator_reports = summary.get("simulator_report_identities")
        if not isinstance(simulator_reports, list) or any(not isinstance(identity, str) for identity in simulator_reports):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} simulator_report_identities must be a string list"
            )
        validate_diagnostic_records(
            summary.get("diagnostic_records"),
            diagnostics,
            f"DSE report bundle runtime evidence summary {index}",
        )
        input_fingerprints = summary.get("input_artifact_fingerprints")
        if not isinstance(input_fingerprints, dict):
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} input_artifact_fingerprints must be an object"
            )
            input_fingerprints = {}
        else:
            for identity, fingerprint in input_fingerprints.items():
                if not isinstance(identity, str) or not identity:
                    diagnostics.append(
                        f"DSE report bundle runtime evidence summary {index} "
                        "input_artifact_fingerprints has invalid identity"
                    )
                    continue
                if not valid_sha256_hex(fingerprint):
                    diagnostics.append(
                        f"DSE report bundle runtime evidence summary {index} "
                        f"input_artifact_fingerprints has invalid fingerprint for {identity}"
                    )
                    continue
                resolved = resolve_artifact_identity_reference(path, identity)
                if resolved is not None and fingerprint != artifact_fingerprint(resolved):
                    diagnostics.append(
                        f"DSE report bundle runtime evidence summary {index} "
                        f"input_artifact_fingerprints stale for {identity!r}"
                    )
        validate_runtime_evidence_input_fingerprint_references(
            input_fingerprints,
            runtime_evidence_artifact_input_references(summary),
            diagnostics,
            f"DSE report bundle runtime evidence summary {index}",
        )
        if require_complete and not input_fingerprints:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} needs input_artifact_fingerprints"
            )
        fallback = summary.get("fallback_decision")
        validate_fallback_decision(
            fallback,
            diagnostics,
            f"DSE report bundle runtime evidence summary {index}",
            expected_policy=summary.get("fallback_policy"),
            target_profile_id=summary.get("target_profile_id"),
            require_complete=True,
        )
        if isinstance(fallback, dict) and fallback.get("decision") == "report_only":
            if summary.get("launch_status") != "not_run" or summary.get("target_status") != "not_run":
                diagnostics.append(
                    f"DSE report bundle runtime evidence summary {index} report_only status must remain not_run"
                )


def validate_dse_report_candidate_input_fingerprints(
    path: Path,
    candidate: dict[str, object],
    diagnostics: list[str],
    index: int,
) -> None:
    input_artifacts = candidate.get("referenced_input_artifacts")
    input_refs = {
        reference
        for reference in input_artifacts
        if isinstance(reference, str) and reference
    } if isinstance(input_artifacts, list) else set()
    input_fingerprints = candidate.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    status = candidate.get("status")
    if status in {"selected", "pareto", "rejected"} and not input_fingerprints:
        diagnostics.append(f"DSE report bundle candidate {index} needs input_artifact_fingerprints")
    for identity, fingerprint in input_fingerprints.items():
        if not isinstance(identity, str) or not identity:
            diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints has invalid identity")
            continue
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(
                f"DSE report bundle candidate {index} input_artifact_fingerprints has invalid fingerprint for {identity}"
            )
            continue
        if identity not in input_refs:
            diagnostics.append(
                f"DSE report bundle candidate {index} input_artifact_fingerprints references {identity!r} outside inputs"
            )
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is not None and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints stale for {identity!r}")
    for reference in input_refs:
        if reference not in input_fingerprints:
            diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints lacks {reference!r}")


def resolve_artifact_identity_reference(anchor: Path, identity: str) -> Path | None:
    for reference in (identity, f"{identity}.json", f"{identity}.csv"):
        resolved = resolve_artifact_reference(anchor, reference)
        if resolved.is_file():
            return resolved
    return None


def validate_dse_report_input_fingerprints(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    reference_ids: set[str] = set()
    for key in (
        "referenced_dse_candidate_artifact_identities",
        "referenced_workload_report_bundle_identities",
        "referenced_hardware_candidate_report_bundle_identities",
    ):
        value = data.get(key)
        if isinstance(value, list):
            reference_ids.update(
                reference
                for reference in value
                if isinstance(reference, str) and reference
            )
    input_fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append("DSE report bundle input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    if data.get("report_status") == "pass" and not input_fingerprints:
        diagnostics.append("DSE report bundle pass needs input_artifact_fingerprints")
    for identity, fingerprint in input_fingerprints.items():
        if not isinstance(identity, str) or not identity:
            diagnostics.append("DSE report bundle input_artifact_fingerprints has invalid identity")
            continue
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(
                f"DSE report bundle input_artifact_fingerprints has invalid fingerprint for {identity}"
            )
            continue
        if identity not in reference_ids:
            diagnostics.append(
                f"DSE report bundle input_artifact_fingerprints references {identity!r} outside report inputs"
            )
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is not None and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"DSE report bundle input_artifact_fingerprints stale for {identity!r}")
    for reference in reference_ids:
        if reference not in input_fingerprints:
            diagnostics.append(f"DSE report bundle input_artifact_fingerprints lacks {reference!r}")


def dse_report_referenced_metric_ids(
    path: Path,
    data: dict[str, object],
) -> set[str]:
    metric_ids: set[str] = set()
    candidate_identities: list[str] = []
    report_identities: list[str] = []
    candidate_references = data.get("referenced_dse_candidate_artifact_identities")
    if isinstance(candidate_references, list):
        candidate_identities.extend(
            identity
            for identity in candidate_references
            if isinstance(identity, str) and identity
        )
    for key in (
        "referenced_workload_report_bundle_identities",
        "referenced_hardware_candidate_report_bundle_identities",
    ):
        identities = data.get(key)
        if isinstance(identities, list):
            report_identities.extend(
                identity
                for identity in identities
                if isinstance(identity, str) and identity
            )
    for identity in candidate_identities:
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is None:
            continue
        try:
            with resolved.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue
        for row in rows:
            parsed_metrics = parse_dse_metric_records(row.get("metric_records", ""), [], 0)
            for name in parsed_metrics:
                metric_id = dse_candidate_metric_id(row, name)
                if metric_id is not None:
                    metric_ids.add(metric_id)
    for identity in report_identities:
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is None:
            continue
        try:
            report = json.loads(resolved.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(report, dict):
            continue
        metrics = report.get("metric_records")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_id = metric.get("metric_id")
            if isinstance(metric_id, str) and metric_id:
                metric_ids.add(metric_id)
    return metric_ids


def dse_report_referenced_candidate_rows(
    path: Path,
    data: dict[str, object],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    references = data.get("referenced_dse_candidate_artifact_identities")
    if not isinstance(references, list):
        return rows
    for identity in references:
        if not isinstance(identity, str) or not identity:
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is None or resolved.suffix != ".csv":
            continue
        try:
            candidate_rows = read_csv_rows(resolved)
        except OSError:
            continue
        rows.extend(candidate_rows)
    return rows


def validate_dse_report_hardware_bundle_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    candidate_hardware = {
        row["hardware"]
        for row in dse_report_referenced_candidate_rows(path, data)
        if valid_identity(row.get("hardware"))
        and row.get("selection_status") in {"selected", "pareto", "rejected"}
    }
    if not candidate_hardware:
        return
    references = data.get("referenced_hardware_candidate_report_bundle_identities")
    if not isinstance(references, list):
        return
    for identity in references:
        if not isinstance(identity, str) or not identity:
            continue
        report = read_resolved_json_reference(path, identity)
        if report is None:
            continue
        if report.get("kind") != "hardware_report_bundle":
            diagnostics.append(f"DSE report bundle hardware report reference {identity!r} has wrong kind")
            continue
        if report.get("report_status") != "pass":
            diagnostics.append(f"DSE report bundle hardware report reference {identity!r} is not passing")
            continue
        if not any(
            hardware_identity_matches(candidate, report.get("hardware_candidate_identity"))
            or hardware_identity_matches(candidate, report.get("fabric_adg_identity"))
            for candidate in candidate_hardware
        ):
            diagnostics.append(
                f"DSE report bundle hardware report reference {identity!r} does not match DSE candidates"
            )


def validate_dse_report_workload_bundle_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    candidate_hardware_by_workload: dict[str, set[str]] = {}
    for row in dse_report_referenced_candidate_rows(path, data):
        if row.get("selection_status") not in {"selected", "pareto", "rejected"}:
            continue
        workload = row.get("workload")
        hardware = row.get("hardware")
        if valid_identity(workload):
            assert workload is not None
            candidate_hardware_by_workload.setdefault(workload, set())
            if valid_identity(hardware):
                assert hardware is not None
                candidate_hardware_by_workload[workload].add(hardware)
    if not candidate_hardware_by_workload:
        return
    references = data.get("referenced_workload_report_bundle_identities")
    if not isinstance(references, list):
        return
    for identity in references:
        if not isinstance(identity, str) or not identity:
            continue
        report = read_resolved_json_reference(path, identity)
        if report is None:
            continue
        if report.get("kind") != "workload_report_bundle":
            diagnostics.append(f"DSE report bundle workload report reference {identity!r} has wrong kind")
            continue
        if report.get("report_status") != "pass":
            diagnostics.append(f"DSE report bundle workload report reference {identity!r} is not passing")
            continue
        workload = report.get("workload")
        if not isinstance(workload, str) or workload not in candidate_hardware_by_workload:
            diagnostics.append(
                f"DSE report bundle workload report reference {identity!r} does not match DSE candidates"
            )
            continue
        candidate_hardware = candidate_hardware_by_workload[workload]
        selected_hardware = report.get("selected_hardware_candidate_identity")
        if candidate_hardware and not any(
            hardware_identity_matches(candidate, selected_hardware)
            for candidate in candidate_hardware
        ):
            diagnostics.append(
                f"DSE report bundle workload report reference {identity!r} selected hardware "
                "does not match DSE candidates"
            )


def validate_dse_report_candidate_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    referenced_status_by_id = {
        row["candidate"]: row.get("selection_status", "")
        for row in dse_report_referenced_candidate_rows(path, data)
        if valid_identity(row.get("candidate"))
        and row.get("selection_status") in {"selected", "pareto", "rejected"}
    }
    if not referenced_status_by_id:
        return
    candidates = data.get("candidate_list")
    if not isinstance(candidates, list):
        return
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        referenced_status = referenced_status_by_id.get(candidate_id)
        if referenced_status is None:
            diagnostics.append(
                f"DSE report bundle candidate {index} is absent from referenced candidate evidence"
            )
            continue
        if candidate.get("status") != referenced_status:
            diagnostics.append(
                f"DSE report bundle candidate {index} status does not match referenced candidate evidence"
            )


def validate_dse_candidate_id_list(
    value: object,
    diagnostics: list[str],
    label: str,
) -> set[str]:
    if not isinstance(value, list):
        return set()
    candidate_ids: set[str] = set()
    repeated: set[str] = set()
    for candidate_id in value:
        if not isinstance(candidate_id, str) or not candidate_id:
            diagnostics.append(f"{label} contains invalid candidate id")
            continue
        if candidate_id in candidate_ids:
            repeated.add(candidate_id)
        candidate_ids.add(candidate_id)
    if repeated:
        diagnostics.append(f"{label} repeats candidate ids {sorted(repeated)}")
    return candidate_ids


def validate_workload_report_input_fingerprints(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    reference_ids: set[str] = set()
    for key in (
        "source_artifact_identity",
        "compiler_command_identity",
        "selected_mapping_artifact_identity",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            reference_ids.add(value)
    optional_identities = data.get("optional_artifact_identities")
    if isinstance(optional_identities, dict):
        reference_ids.update(
            identity
            for identity in optional_identities.values()
            if isinstance(identity, str) and identity
        )
    metrics = data.get("metric_records")
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            source = metric.get("evidence_source_artifact_id")
            if isinstance(source, str) and source:
                reference_ids.add(source)
    input_fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append("workload report bundle input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    if data.get("report_status") == "pass" and not input_fingerprints:
        diagnostics.append("workload report bundle pass needs input_artifact_fingerprints")
    for identity, fingerprint in input_fingerprints.items():
        if not isinstance(identity, str) or not identity:
            diagnostics.append("workload report bundle input_artifact_fingerprints has invalid identity")
            continue
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(
                f"workload report bundle input_artifact_fingerprints has invalid fingerprint for {identity}"
            )
            continue
        if identity not in reference_ids:
            diagnostics.append(
                f"workload report bundle input_artifact_fingerprints references {identity!r} outside report inputs"
            )
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is not None and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"workload report bundle input_artifact_fingerprints stale for {identity!r}")
    for reference in reference_ids:
        if reference not in input_fingerprints:
            diagnostics.append(f"workload report bundle input_artifact_fingerprints lacks {reference!r}")


def validate_workload_runtime_evidence_references(
    data: dict[str, object],
    runtime_evidence: object,
    diagnostics: list[str],
) -> None:
    if not isinstance(runtime_evidence, dict):
        return
    optional_identities = data.get("optional_artifact_identities")
    runtime_package_identity = None
    if isinstance(optional_identities, dict):
        runtime_package_identity = optional_identities.get("runtime_package")
    evidence_runtime_package_identity = runtime_evidence.get("runtime_package_identity")
    if isinstance(evidence_runtime_package_identity, str) and evidence_runtime_package_identity:
        if runtime_package_identity != evidence_runtime_package_identity:
            diagnostics.append(
                "workload report bundle runtime_evidence runtime_package_identity "
                "does not match runtime package input"
            )
    runtime_fallback_decision = data.get("runtime_fallback_decision")
    evidence_fallback_decision = runtime_evidence.get("fallback_decision")
    if (
        isinstance(runtime_fallback_decision, dict)
        and runtime_fallback_decision
        and isinstance(evidence_fallback_decision, dict)
        and evidence_fallback_decision
        and runtime_fallback_decision != evidence_fallback_decision
    ):
        diagnostics.append(
            "workload report bundle runtime_fallback_decision does not match runtime_evidence"
        )
    runtime_host_interface = data.get("runtime_host_interface")
    evidence_host_interface = runtime_evidence.get("host_interface")
    if (
        isinstance(runtime_host_interface, dict)
        and runtime_host_interface
        and isinstance(evidence_host_interface, dict)
        and evidence_host_interface
        and runtime_host_interface != evidence_host_interface
    ):
        diagnostics.append(
            "workload report bundle runtime_host_interface does not match runtime_evidence"
        )
    selected_hardware = data.get("selected_hardware_candidate_identity")
    evidence_fabric = runtime_evidence.get("fabric_adg_identity")
    if (
        isinstance(selected_hardware, str)
        and selected_hardware
        and isinstance(evidence_fabric, str)
        and evidence_fabric
        and selected_hardware != evidence_fabric
    ):
        diagnostics.append(
            "workload report bundle runtime_evidence fabric_adg_identity "
            "does not match selected hardware candidate"
        )
    runtime_input_identity = data.get("runtime_input_identity")
    if isinstance(runtime_input_identity, str) and runtime_input_identity:
        work_package_metadata = runtime_evidence.get("work_package_metadata")
        if isinstance(work_package_metadata, dict):
            if work_package_metadata.get("runtime_input_identity") != runtime_input_identity:
                diagnostics.append(
                    "workload report bundle runtime_evidence work_package_metadata "
                    "runtime_input_identity does not match runtime input"
                )
        host_interface = runtime_evidence.get("host_interface")
        if isinstance(host_interface, dict):
            if host_interface.get("source_provenance") != runtime_input_identity:
                diagnostics.append(
                    "workload report bundle runtime_evidence host_interface "
                    "source_provenance does not match runtime input"
                )
    selected_mapping = data.get("selected_mapping_artifact_identity")
    if isinstance(selected_mapping, str) and selected_mapping:
        if runtime_evidence.get("mapping_artifact_identity") != selected_mapping:
            diagnostics.append(
                "workload report bundle runtime_evidence mapping_artifact_identity "
                "does not match selected mapping artifact"
            )


def validate_workload_runtime_package_projection(
    path: Path,
    data: dict[str, object],
    runtime_evidence: object,
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass" or not isinstance(runtime_evidence, dict):
        return
    optional_identities = data.get("optional_artifact_identities")
    if not isinstance(optional_identities, dict):
        return
    runtime_package_identity = optional_identities.get("runtime_package")
    package = read_resolved_json_reference(path, runtime_package_identity)
    if package is None:
        return
    if package.get("kind") != "runtime_package":
        diagnostics.append("workload report bundle runtime package reference has wrong kind")
        return
    if package.get("status") != "pass":
        diagnostics.append("workload report bundle runtime package reference is not passing")
        return
    expected = runtime_evidence_helpers.runtime_evidence_from_package(
        package,
        runtime_package_identity if isinstance(runtime_package_identity, str) else "",
    )
    for key, expected_value in expected.items():
        if runtime_evidence.get(key) != expected_value:
            diagnostics.append(
                f"workload report bundle runtime_evidence {key} does not match referenced runtime package"
            )


def validate_hardware_report_input_fingerprints(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    reference_ids: set[str] = set()
    rtl_manifest_identity = data.get("rtl_manifest_identity")
    if isinstance(rtl_manifest_identity, str) and rtl_manifest_identity:
        reference_ids.add(rtl_manifest_identity)
    for key in ("eda_report_identities", "fpa_report_identities"):
        value = data.get(key)
        if isinstance(value, list):
            reference_ids.update(
                identity
                for identity in value
                if isinstance(identity, str) and identity
            )
    metrics = data.get("metric_records")
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            source = metric.get("evidence_source_artifact_id")
            if isinstance(source, str) and source:
                reference_ids.add(source)
    input_fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append("hardware report bundle input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    if data.get("report_status") == "pass" and not input_fingerprints:
        diagnostics.append("hardware report bundle pass needs input_artifact_fingerprints")
    for identity, fingerprint in input_fingerprints.items():
        if not isinstance(identity, str) or not identity:
            diagnostics.append("hardware report bundle input_artifact_fingerprints has invalid identity")
            continue
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(
                f"hardware report bundle input_artifact_fingerprints has invalid fingerprint for {identity}"
            )
            continue
        if identity not in reference_ids:
            diagnostics.append(
                f"hardware report bundle input_artifact_fingerprints references {identity!r} outside report inputs"
            )
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is not None and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"hardware report bundle input_artifact_fingerprints stale for {identity!r}")
    for reference in reference_ids:
        if reference not in input_fingerprints:
            diagnostics.append(f"hardware report bundle input_artifact_fingerprints lacks {reference!r}")


def hardware_identity_matches(candidate: object, hardware: object) -> bool:
    if not isinstance(candidate, str) or not isinstance(hardware, str):
        return False
    if not candidate or not hardware:
        return False
    return (
        candidate == hardware
        or candidate.rsplit("::", 1)[-1] == hardware
        or hardware.rsplit("::", 1)[-1] == candidate
    )


def validate_hardware_report_fpa_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    hardware = data.get("hardware_candidate_identity")
    fpa_identities = data.get("fpa_report_identities")
    if not isinstance(fpa_identities, list):
        return
    matching_workloads: set[str] = set()
    for identity in fpa_identities:
        if not isinstance(identity, str) or not identity:
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is None or resolved.suffix != ".csv":
            continue
        rows = read_csv_rows(resolved)
        matching_rows = [
            row
            for row in rows
            if row.get("status") == "pass"
            and hardware_identity_matches(row.get("hardware"), hardware)
        ]
        if not matching_rows:
            diagnostics.append(
                f"hardware report bundle FPA reference {identity!r} does not match hardware candidate"
            )
            continue
        matching_workloads.update(
            row["workload"]
            for row in matching_rows
            if isinstance(row.get("workload"), str) and row.get("workload")
        )
    supported_workloads = data.get("supported_workload_classes")
    if isinstance(supported_workloads, list) and matching_workloads:
        missing_workloads = sorted(
            workload
            for workload in supported_workloads
            if isinstance(workload, str) and workload and workload not in matching_workloads
        )
        if missing_workloads:
            diagnostics.append(
                "hardware report bundle supported workloads are absent from FPA evidence "
                f"{missing_workloads}"
            )


def validate_hardware_report_adg_builder_recipe_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    input_fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict) or "adg-hardware-summary" not in input_fingerprints:
        return
    resolved = resolve_artifact_identity_reference(path, "adg-hardware-summary")
    if resolved is None or resolved.suffix != ".csv":
        return
    hardware = data.get("hardware_candidate_identity")
    matching_rows = [
        row
        for row in read_csv_rows(resolved)
        if row.get("verify_status") == "pass"
        and hardware_identity_matches(row.get("hardware"), hardware)
    ]
    if not matching_rows:
        return
    expected = matching_rows[0].get("adg_builder_recipe_identity", "")
    if data.get("adg_builder_recipe_identity") != expected:
        diagnostics.append(
            "hardware report bundle ADG builder recipe identity does not match ADG hardware summary"
        )


def validate_hardware_report_rtl_manifest_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    manifest = read_resolved_json_reference(path, data.get("rtl_manifest_identity"))
    if manifest is None:
        return
    if manifest.get("kind") != "rtl_manifest":
        diagnostics.append("hardware report bundle RTL manifest reference has wrong kind")
        return
    if manifest.get("status") != "pass":
        diagnostics.append("hardware report bundle RTL manifest reference is not passing")
    source_fabric = manifest.get("source_fabric_adg_identity")
    if not (
        hardware_identity_matches(source_fabric, data.get("fabric_adg_identity"))
        or hardware_identity_matches(source_fabric, data.get("hardware_candidate_identity"))
    ):
        diagnostics.append("hardware report bundle RTL manifest source does not match hardware candidate")


def validate_hardware_report_eda_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    rtl_manifest_identity = data.get("rtl_manifest_identity")
    eda_identities = data.get("eda_report_identities")
    if not isinstance(eda_identities, list):
        return
    for identity in eda_identities:
        if not isinstance(identity, str) or not identity:
            continue
        report = read_resolved_json_reference(path, identity)
        if report is None:
            continue
        if report.get("kind") != "eda_report":
            diagnostics.append(f"hardware report bundle EDA reference {identity!r} has wrong kind")
            continue
        if report.get("status") != "pass":
            diagnostics.append(f"hardware report bundle EDA reference {identity!r} is not passing")
            continue
        if report.get("capability_class") != "rtl_lint":
            diagnostics.append(f"hardware report bundle EDA reference {identity!r} has wrong capability class")
            continue
        if report.get("rtl_manifest_identity") != rtl_manifest_identity:
            diagnostics.append(
                f"hardware report bundle EDA reference {identity!r} does not match RTL manifest"
            )


def validate_eda_report_input_fingerprints(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    reference_ids: set[str] = set()
    rtl_manifest_identity = data.get("rtl_manifest_identity")
    if isinstance(rtl_manifest_identity, str) and rtl_manifest_identity:
        reference_ids.add(rtl_manifest_identity)
    input_fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict):
        diagnostics.append("EDA report input_artifact_fingerprints must be an object")
        input_fingerprints = {}
    if data.get("status") in {"pass", "fail", "blocked"} and not input_fingerprints:
        diagnostics.append("EDA report needs input_artifact_fingerprints")
    for identity, fingerprint in input_fingerprints.items():
        if not isinstance(identity, str) or not identity:
            diagnostics.append("EDA report input_artifact_fingerprints has invalid identity")
            continue
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(f"EDA report input_artifact_fingerprints has invalid fingerprint for {identity}")
            continue
        if identity not in reference_ids:
            diagnostics.append(
                f"EDA report input_artifact_fingerprints references {identity!r} outside report inputs"
            )
            continue
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is not None and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"EDA report input_artifact_fingerprints stale for {identity!r}")
    for reference in reference_ids:
        if reference not in input_fingerprints:
            diagnostics.append(f"EDA report input_artifact_fingerprints lacks {reference!r}")


def validate_eda_report_source_fingerprints(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    source_files = data.get("checked_source_files")
    source_fingerprints = data.get("source_file_fingerprints")
    if not isinstance(source_files, list):
        diagnostics.append("EDA report checked_source_files must be a list")
        source_files = []
    if not isinstance(source_fingerprints, dict):
        diagnostics.append("EDA report source_file_fingerprints must be an object")
        source_fingerprints = {}
    for source in source_files:
        if not isinstance(source, str) or not source:
            diagnostics.append("EDA report checked_source_files contains invalid path")
            continue
        fingerprint = source_fingerprints.get(source)
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(f"EDA report source_file_fingerprints has invalid fingerprint for {source!r}")
            continue
        resolved = resolve_artifact_reference(path, source)
        if not resolved.is_file():
            diagnostics.append(f"EDA report checked source {source!r} does not exist")
            continue
        if fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"EDA report source_file_fingerprints stale for {source!r}")
    for source in source_fingerprints:
        if source not in source_files:
            diagnostics.append(f"EDA report source_file_fingerprints references unchecked source {source!r}")


def validate_eda_report_rtl_manifest_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    manifest = read_resolved_json_reference(path, data.get("rtl_manifest_identity"))
    if manifest is None:
        return
    if manifest.get("kind") != "rtl_manifest":
        diagnostics.append("EDA report RTL manifest reference has wrong kind")
        return
    if data.get("status") == "pass" and manifest.get("status") != "pass":
        diagnostics.append("EDA report pass references a non-passing RTL manifest")
    manifest_tops = manifest.get("top_level_modules")
    checked_tops = data.get("checked_top_modules")
    if isinstance(manifest_tops, list) and isinstance(checked_tops, list):
        missing = [
            top
            for top in manifest_tops
            if isinstance(top, str) and top and top not in checked_tops
        ]
        if data.get("status") == "pass" and missing:
            diagnostics.append(f"EDA report pass missed RTL manifest top modules {missing}")


def read_resolved_json_reference(path: Path, identity: object) -> dict[str, object] | None:
    if not isinstance(identity, str) or not identity:
        return None
    resolved = resolve_artifact_identity_reference(path, identity)
    if resolved is None:
        return None
    try:
        value = json.loads(resolved.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_workload_graph_set_aggregate(data: dict[str, object]) -> bool:
    return data.get("aggregation_kind") == "workload_graph_set"


def aggregate_component_identities(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [identity for identity in value if isinstance(identity, str) and identity]


def aggregate_component_reports(
    path: Path,
    data: dict[str, object],
    key: str,
    expected_kind: str,
    diagnostics: list[str],
    label: str,
) -> list[dict[str, object]]:
    identities = aggregate_component_identities(data, key)
    if not identities:
        diagnostics.append(f"{label} needs non-empty {key}")
        return []
    reports: list[dict[str, object]] = []
    fingerprints = data.get("input_artifact_fingerprints")
    if not isinstance(fingerprints, dict):
        diagnostics.append(f"{label} input_artifact_fingerprints must be an object")
        fingerprints = {}
    for identity in identities:
        resolved = resolve_artifact_identity_reference(path, identity)
        if resolved is None:
            diagnostics.append(f"{label} component reference {identity!r} does not resolve")
            continue
        try:
            report = json.loads(resolved.read_text())
        except json.JSONDecodeError:
            diagnostics.append(f"{label} component reference {identity!r} is not JSON")
            continue
        if not isinstance(report, dict) or json_kind_for_path(resolved) != expected_kind:
            diagnostics.append(f"{label} component reference {identity!r} has wrong artifact kind")
            continue
        if report.get("status") != "pass":
            diagnostics.append(f"{label} component reference {identity!r} is not passing")
        fingerprint = fingerprints.get(identity)
        if not valid_sha256_hex(fingerprint):
            diagnostics.append(f"{label} input_artifact_fingerprints lacks valid fingerprint for {identity!r}")
        elif fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"{label} input_artifact_fingerprints stale for {identity!r}")
        reports.append(report)
    return reports


def prefer_workload_graph_set_aggregates(
    grouped: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    preferred: dict[str, list[dict[str, object]]] = {}
    for workload, reports in grouped.items():
        aggregates = [report for report in reports if is_workload_graph_set_aggregate(report)]
        preferred[workload] = aggregates if aggregates else reports
    return preferred


def simulator_report_runtime_input_identity(report: dict[str, object], workload: object) -> str:
    identity = report.get("runtime_input_identity")
    if isinstance(identity, str) and identity:
        return identity
    if isinstance(workload, str) and workload:
        return f"test-app-fixture::{workload}::default"
    return ""


def validate_sim_comparison_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
    label: str = "simulation comparison report",
) -> None:
    workload = data.get("workload")
    runtime_input_identity = data.get("runtime_input_identity")
    dfg_report = read_resolved_json_reference(path, data.get("dfg_sim_report_identity"))
    if dfg_report is not None:
        if dfg_report.get("kind") != "dfg_sim_report":
            diagnostics.append(f"{label} DFG reference is not a DFG simulator report")
        if isinstance(workload, str) and workload and dfg_report.get("workload") != workload:
            diagnostics.append(f"{label} DFG reference workload does not match report")
        expected_runtime_input = simulator_report_runtime_input_identity(dfg_report, workload)
        if (
            isinstance(runtime_input_identity, str)
            and runtime_input_identity
            and expected_runtime_input
            and expected_runtime_input != runtime_input_identity
        ):
            diagnostics.append(f"{label} DFG reference runtime input does not match report")

    cgra_report = read_resolved_json_reference(path, data.get("cgra_sim_report_identity"))
    if cgra_report is not None:
        if cgra_report.get("kind") != "cgra_sim_report":
            diagnostics.append(f"{label} CGRA reference is not a CGRA simulator report")
        if isinstance(workload, str) and workload and cgra_report.get("workload") != workload:
            diagnostics.append(f"{label} CGRA reference workload does not match report")
        expected_runtime_input = simulator_report_runtime_input_identity(cgra_report, workload)
        if (
            isinstance(runtime_input_identity, str)
            and runtime_input_identity
            and expected_runtime_input
            and expected_runtime_input != runtime_input_identity
        ):
            diagnostics.append(f"{label} CGRA reference runtime input does not match report")

    mapping_report = read_resolved_json_reference(path, data.get("mapping_artifact_identity"))
    if mapping_report is not None:
        if mapping_report.get("kind") != "pnr_mapping":
            diagnostics.append(f"{label} mapping reference is not a PnR mapping artifact")
        if isinstance(workload, str) and workload and mapping_report.get("workload") != workload:
            diagnostics.append(f"{label} mapping reference workload does not match report")
        if cgra_report is not None:
            mapping_id = mapping_report.get("mapping_id")
            cgra_mapping_id = cgra_report.get("mapping_id")
            if (
                isinstance(mapping_id, str)
                and mapping_id
                and isinstance(cgra_mapping_id, str)
                and cgra_mapping_id
                and mapping_id != cgra_mapping_id
            ):
                diagnostics.append(f"{label} mapping reference does not match CGRA report")


def validate_workload_report_sim_comparison_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("report_status") != "pass":
        return
    optional_identities = data.get("optional_artifact_identities")
    if not isinstance(optional_identities, dict):
        return
    comparison_identity = optional_identities.get("simulation_comparison_report")
    comparison = read_resolved_json_reference(path, comparison_identity)
    if comparison is None:
        return
    if comparison.get("kind") != "sim_comparison_report":
        diagnostics.append("workload report bundle simulation comparison reference has wrong kind")
        return
    validate_sim_comparison_references(
        path,
        comparison,
        diagnostics,
        "workload report bundle simulation comparison",
    )
    for comparison_key, bundle_key, label in (
        ("workload", "workload", "workload"),
        ("runtime_input_identity", "runtime_input_identity", "runtime input"),
        ("mapping_artifact_identity", "selected_mapping_artifact_identity", "mapping artifact"),
    ):
        comparison_value = comparison.get(comparison_key)
        bundle_value = data.get(bundle_key)
        if (
            isinstance(comparison_value, str)
            and comparison_value
            and isinstance(bundle_value, str)
            and bundle_value
            and comparison_value != bundle_value
        ):
            diagnostics.append(
                f"workload report bundle simulation comparison {label} does not match bundle"
            )
    expected_identities: dict[str, object] = {}
    expected_identities["dfg_sim_report_identity"] = optional_identities.get("dfg_sim_report")
    expected_identities["cgra_sim_report_identity"] = optional_identities.get("cgra_sim_report")
    for key, expected in expected_identities.items():
        actual = comparison.get(key)
        if (
            isinstance(actual, str)
            and actual
            and isinstance(expected, str)
            and expected
            and actual != expected
        ):
            diagnostics.append(
                f"workload report bundle simulation comparison {key} does not match bundle input"
            )


def validate_runtime_package_sim_comparison_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("status") != "pass":
        return
    simulator_reports = data.get("simulator_report_identities")
    if not isinstance(simulator_reports, list):
        return
    comparison_identity = next(
        (
            identity
            for identity in simulator_reports
            if isinstance(identity, str) and identity.endswith("sim-comparison-report")
        ),
        "",
    )
    comparison = read_resolved_json_reference(path, comparison_identity)
    if comparison is None:
        return
    if comparison.get("kind") != "sim_comparison_report":
        diagnostics.append("runtime package simulation comparison reference has wrong kind")
        return
    validate_sim_comparison_references(
        path,
        comparison,
        diagnostics,
        "runtime package simulation comparison",
    )
    work_package_metadata = data.get("work_package_metadata")
    runtime_input_identity = None
    if isinstance(work_package_metadata, dict):
        runtime_input_identity = work_package_metadata.get("runtime_input_identity")
    if not isinstance(runtime_input_identity, str) or not runtime_input_identity:
        runtime_input_identity = runtime_source_provenance(data)
    for comparison_key, package_value, label in (
        ("workload", data.get("workload"), "workload"),
        ("runtime_input_identity", runtime_input_identity, "runtime input"),
        ("mapping_artifact_identity", data.get("selected_mapping_artifact_identity"), "mapping artifact"),
    ):
        comparison_value = comparison.get(comparison_key)
        if (
            isinstance(comparison_value, str)
            and comparison_value
            and isinstance(package_value, str)
            and package_value
            and comparison_value != package_value
        ):
            diagnostics.append(
                f"runtime package simulation comparison {label} does not match package"
            )
    cgra_identity = next(
        (
            identity
            for identity in simulator_reports
            if isinstance(identity, str) and identity.endswith("cgra-sim-report")
        ),
        "",
    )
    comparison_cgra_identity = comparison.get("cgra_sim_report_identity")
    if (
        isinstance(comparison_cgra_identity, str)
        and comparison_cgra_identity
        and isinstance(cgra_identity, str)
        and cgra_identity
        and comparison_cgra_identity != cgra_identity
    ):
        diagnostics.append("runtime package simulation comparison CGRA report does not match package")


def validate_runtime_package_simulator_report_references(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("status") != "pass":
        return
    simulator_reports = data.get("simulator_report_identities")
    if not isinstance(simulator_reports, list):
        return
    for identity in simulator_reports:
        if not isinstance(identity, str) or not identity:
            continue
        report = read_resolved_json_reference(path, identity)
        if report is None:
            continue
        kind = report.get("kind")
        if kind == "cgra_sim_report":
            if report.get("status") != "pass":
                diagnostics.append(f"runtime package simulator report {identity!r} is not passing")
            if report.get("workload") != data.get("workload"):
                diagnostics.append(f"runtime package simulator report {identity!r} workload does not match package")
            if not hardware_identity_matches(report.get("hardware"), data.get("fabric_adg_identity")):
                diagnostics.append(f"runtime package simulator report {identity!r} hardware does not match package")
            if not data.get("selected_mapping_artifact_identity"):
                diagnostics.append("runtime package CGRA simulator report requires mapping artifact identity")
            mapping = read_resolved_json_reference(path, data.get("selected_mapping_artifact_identity"))
            if mapping is not None:
                if mapping.get("kind") != "pnr_mapping":
                    diagnostics.append("runtime package selected mapping reference has wrong kind")
                elif report.get("mapping_id") != mapping.get("mapping_id"):
                    diagnostics.append(f"runtime package simulator report {identity!r} mapping does not match package")
        elif kind == "dfg_sim_report":
            if report.get("status") != "pass":
                diagnostics.append(f"runtime package simulator report {identity!r} is not passing")
            if report.get("workload") != data.get("workload"):
                diagnostics.append(f"runtime package simulator report {identity!r} workload does not match package")
            if data.get("selected_mapping_artifact_identity"):
                diagnostics.append("runtime package DFG simulator report must not require mapping artifact identity")
        elif kind == "sim_comparison_report":
            continue
        else:
            diagnostics.append(f"runtime package simulator report {identity!r} has wrong kind")


def validate_runtime_package_mapping_reference(
    path: Path,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    if data.get("status") != "pass":
        return
    mapping_identity = data.get("selected_mapping_artifact_identity")
    if not isinstance(mapping_identity, str) or not mapping_identity:
        return
    mapping = read_resolved_json_reference(path, mapping_identity)
    if mapping is None:
        return
    if mapping.get("kind") != "pnr_mapping":
        diagnostics.append("runtime package selected mapping reference has wrong kind")
        return
    if mapping.get("status") != "pass":
        diagnostics.append("runtime package selected mapping reference is not passing")
    if mapping.get("workload") != data.get("workload"):
        diagnostics.append("runtime package selected mapping workload does not match package")
    if not hardware_identity_matches(mapping.get("hardware"), data.get("fabric_adg_identity")):
        diagnostics.append("runtime package selected mapping hardware does not match package")


def audit_json(path: Path, kind: str) -> dict[str, object]:
    diagnostics: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "artifact": str(path),
            "schema": kind,
            "entries_checked": 0,
            "parser_checks": ["json_parse"],
            "finding": "fail",
            "diagnostics": [str(exc)],
        }
    required = JSON_SCHEMAS[kind]["required_keys"]
    assert isinstance(required, set)
    missing = sorted(required - set(data))
    if missing:
        diagnostics.append(f"missing keys {missing}")
    if data.get("schema_version") != 1:
        diagnostics.append("schema_version must be 1")
    if kind == "artifact_manifest" and data.get("diagnostics"):
        diagnostics.append("artifact manifest contains blocked diagnostics")
    manifest_entries_checked: int | None = None
    if kind == "artifact_manifest" and isinstance(data, dict):
        manifest_entries_checked = validate_artifact_manifest_edges(data, diagnostics, path)
    if kind == "artifact_audit" and data.get("verdict") not in {"pass", "fail"}:
        diagnostics.append("artifact audit verdict must be pass or fail")
    if kind == "rtl_manifest":
        if data.get("kind") != "rtl_manifest":
            diagnostics.append("RTL manifest kind must be rtl_manifest")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("RTL manifest status must be a known status")
        for key in ("manifest_id", "source_fabric_adg_identity", "mapping_artifact_identity"):
            if not isinstance(data.get(key), str):
                diagnostics.append(f"RTL manifest {key} must be a string")
        if not isinstance(data.get("lowering_configuration"), dict):
            diagnostics.append("RTL manifest lowering_configuration must be an object")
        for key in (
            "emitted_source_files",
            "top_level_modules",
            "generated_packages",
            "generated_interfaces",
            "black_box_modules",
            "behavioral_models",
            "required_tool_capability_classes",
            "required_library_profile_classes",
            "constraints",
            "activity_hooks",
            "diagnostics",
        ):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"RTL manifest {key} must be a list")
        sources = data.get("emitted_source_files")
        if not isinstance(sources, list):
            sources = []
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                diagnostics.append(f"RTL manifest source {index} must be an object")
                continue
            source_path_raw = source.get("path")
            if not isinstance(source_path_raw, str) or not source_path_raw:
                diagnostics.append(f"RTL manifest source {index} lacks path")
                continue
            source_path = Path(source_path_raw)
            if source_path.is_absolute():
                diagnostics.append(f"RTL manifest source {index} path must be relative")
                continue
            resolved_source = path.parent / source_path
            if not resolved_source.is_file():
                diagnostics.append(f"RTL manifest source {index} path does not exist")
                continue
            if source.get("language") != "systemverilog":
                diagnostics.append(f"RTL manifest source {index} language must be systemverilog")
            fingerprint = source.get("fingerprint")
            if not valid_sha256_hex(fingerprint):
                diagnostics.append(f"RTL manifest source {index} has invalid fingerprint")
            elif fingerprint != artifact_fingerprint(resolved_source):
                diagnostics.append(f"RTL manifest source {index} fingerprint is stale")
        if data.get("status") == "pass":
            if not data.get("source_fabric_adg_identity"):
                diagnostics.append("RTL manifest pass needs source_fabric_adg_identity")
            if not sources:
                diagnostics.append("RTL manifest pass needs emitted_source_files")
            if not data.get("top_level_modules"):
                diagnostics.append("RTL manifest pass needs top_level_modules")
            if not data.get("required_tool_capability_classes"):
                diagnostics.append("RTL manifest pass needs required_tool_capability_classes")
    if kind == "dfg_sim_report":
        if data.get("kind") != "dfg_sim_report":
            diagnostics.append("DFG simulator report kind must be dfg_sim_report")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("DFG simulator report status must be a known status")
        if data.get("metric_definition") != "optimistic_pipeline_latency_throughput_sum":
            diagnostics.append("DFG simulator report has unknown metric definition")
        if data.get("operation_semantics_source") != "loom.sim.operation_semantics.v1":
            diagnostics.append("DFG simulator report has unknown operation semantics source")
        if data.get("operation_cost_model_source") != "loom.sim.operation_cost.v1":
            diagnostics.append("DFG simulator report has unknown operation cost model source")
        cycles = data.get("optimistic_cycles")
        if not isinstance(cycles, int) or cycles < 0:
            diagnostics.append("DFG simulator report optimistic_cycles must be non-negative integer")
        event_count = data.get("event_count")
        if not isinstance(event_count, int) or event_count < 0:
            diagnostics.append("DFG simulator report event_count must be non-negative integer")
        wavefront_steps = data.get("wavefront_steps")
        if not isinstance(wavefront_steps, int) or wavefront_steps < 0:
            diagnostics.append("DFG simulator report wavefront_steps must be non-negative integer")
        dynamic_work_items = data.get("dynamic_work_items")
        if not isinstance(dynamic_work_items, int) or dynamic_work_items < 0:
            diagnostics.append("DFG simulator report dynamic_work_items must be non-negative integer")
        operation_fire_counts = data.get("operation_fire_counts")
        if not isinstance(operation_fire_counts, dict) or not operation_fire_counts:
            diagnostics.append("DFG simulator report needs non-empty operation_fire_counts")
        elif not all(
            isinstance(name, str) and isinstance(count, int) and count >= 0
            for name, count in operation_fire_counts.items()
        ):
            diagnostics.append("DFG simulator report operation_fire_counts must map op names to non-negative integers")
        if isinstance(cycles, int) and isinstance(event_count, int) and cycles < event_count:
            diagnostics.append("DFG simulator optimistic_cycles must not be below event_count")
        if is_workload_graph_set_aggregate(data):
            components = aggregate_component_reports(
                path,
                data,
                "component_dfg_sim_report_identities",
                "dfg_sim_report",
                diagnostics,
                "DFG simulator aggregate report",
            )
            if components:
                workload = data.get("workload")
                if any(component.get("workload") != workload for component in components):
                    diagnostics.append("DFG simulator aggregate components have mismatched workload")
                component_graphs = {
                    component.get("graph")
                    for component in components
                    if isinstance(component.get("graph"), str)
                }
                declared_graphs = set(aggregate_component_identities(data, "component_graphs"))
                if declared_graphs and declared_graphs != component_graphs:
                    diagnostics.append("DFG simulator aggregate component_graphs do not match components")
                for key in ("optimistic_cycles", "wavefront_steps", "event_count", "dynamic_work_items"):
                    value = data.get(key)
                    component_sum = sum(
                        int(component[key])
                        for component in components
                        if isinstance(component.get(key), int)
                    )
                    if isinstance(value, int) and value != component_sum:
                        diagnostics.append(f"DFG simulator aggregate {key} does not match component sum")
    if kind == "cgra_sim_report":
        if data.get("kind") != "cgra_sim_report":
            diagnostics.append("CGRA simulator report kind must be cgra_sim_report")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("CGRA simulator report status must be a known status")
        if data.get("fidelity_level") != "mapping_constraint_estimate":
            diagnostics.append("CGRA simulator report has unknown fidelity level")
        if data.get("metric_definition") != "mapping_constraint_estimate":
            diagnostics.append("CGRA simulator report has unknown metric definition")
        if data.get("operation_semantics_source") != "loom.sim.operation_semantics.v1":
            diagnostics.append("CGRA simulator report has unknown operation semantics source")
        if data.get("operation_cost_model_source") != "loom.sim.operation_cost.v1":
            diagnostics.append("CGRA simulator report has unknown operation cost model source")
        if data.get("hardware_bound_classification") != "within_modeled_bounds":
            diagnostics.append("CGRA simulator report has unknown hardware bound classification")
        difference = data.get("difference_classification")
        dfg_cycles = data.get("dfg_cycles")
        hardware_cycles = data.get("hardware_aware_cycles")
        lower_bound = data.get("modeled_lower_bound_cycles")
        delta = data.get("performance_delta_cycles")
        route_cycles = data.get("route_latency_cycles")
        memory_cycles = data.get("memory_latency_cycles")
        temporal_cycles = data.get("temporal_penalty_cycles")
        if not isinstance(dfg_cycles, int) or dfg_cycles < 0:
            diagnostics.append("CGRA simulator report dfg_cycles must be non-negative integer")
        if not isinstance(hardware_cycles, int) or hardware_cycles < 0:
            diagnostics.append("CGRA simulator report hardware_aware_cycles must be non-negative integer")
        if not isinstance(lower_bound, int) or lower_bound < 0:
            diagnostics.append("CGRA simulator report modeled_lower_bound_cycles must be non-negative integer")
        if not isinstance(delta, int) or delta < 0:
            diagnostics.append("CGRA simulator report performance_delta_cycles must be non-negative integer")
        if not isinstance(route_cycles, int) or route_cycles < 0:
            diagnostics.append("CGRA simulator report route_latency_cycles must be non-negative integer")
        if not isinstance(memory_cycles, int) or memory_cycles < 0:
            diagnostics.append("CGRA simulator report memory_latency_cycles must be non-negative integer")
        if not isinstance(temporal_cycles, int) or temporal_cycles < 0:
            diagnostics.append("CGRA simulator report temporal_penalty_cycles must be non-negative integer")
        if isinstance(dfg_cycles, int) and isinstance(hardware_cycles, int) and hardware_cycles < dfg_cycles:
            diagnostics.append("CGRA simulator report is more optimistic than DFG-sim")
        if isinstance(lower_bound, int) and isinstance(hardware_cycles, int) and hardware_cycles < lower_bound:
            diagnostics.append("CGRA simulator report violates modeled lower bound")
        if isinstance(dfg_cycles, int) and isinstance(hardware_cycles, int) and isinstance(delta, int):
            if hardware_cycles - dfg_cycles != delta:
                diagnostics.append("CGRA simulator report delta does not match hardware minus DFG cycles")
        if isinstance(delta, int):
            if delta == 0 and difference != "no_modeled_hardware_constraints":
                diagnostics.append("CGRA simulator zero-delta report needs no-constraint classification")
            if delta > 0 and difference != "expected_hardware_constraint":
                diagnostics.append("CGRA simulator positive-delta report needs hardware-constraint classification")
        if all(
            isinstance(value, int)
            for value in (route_cycles, memory_cycles, temporal_cycles, delta)
        ):
            if route_cycles + memory_cycles + temporal_cycles != delta:
                diagnostics.append("CGRA simulator report delta is not explained by route, memory, and temporal cycles")
        breakdown = data.get("cycle_breakdown")
        if not isinstance(breakdown, list) or not breakdown:
            diagnostics.append("CGRA simulator report needs non-empty cycle_breakdown")
        constraints = data.get("unmodeled_constraints")
        if not isinstance(constraints, list):
            diagnostics.append("CGRA simulator report needs unmodeled_constraints list")
        checks = data.get("first_principles_checks")
        if not isinstance(checks, list) or not checks:
            diagnostics.append("CGRA simulator report needs first_principles_checks")
        elif any(not isinstance(check, dict) or check.get("status") != "pass" for check in checks):
            diagnostics.append("CGRA simulator report has failing first-principles check")
        if is_workload_graph_set_aggregate(data):
            dfg_components = aggregate_component_reports(
                path,
                data,
                "component_dfg_sim_report_identities",
                "dfg_sim_report",
                diagnostics,
                "CGRA simulator aggregate report",
            )
            if dfg_components:
                workload = data.get("workload")
                if any(component.get("workload") != workload for component in dfg_components):
                    diagnostics.append("CGRA simulator aggregate DFG components have mismatched workload")
                component_dfg_cycles = sum(
                    int(component["optimistic_cycles"])
                    for component in dfg_components
                    if isinstance(component.get("optimistic_cycles"), int)
                )
                if isinstance(dfg_cycles, int) and dfg_cycles != component_dfg_cycles:
                    diagnostics.append("CGRA simulator aggregate dfg_cycles do not match DFG component sum")
            components = aggregate_component_reports(
                path,
                data,
                "component_cgra_sim_report_identities",
                "cgra_sim_report",
                diagnostics,
                "CGRA simulator aggregate report",
            )
            if components:
                workload = data.get("workload")
                hardware = data.get("hardware")
                if any(component.get("workload") != workload for component in components):
                    diagnostics.append("CGRA simulator aggregate components have mismatched workload")
                if any(component.get("hardware") != hardware for component in components):
                    diagnostics.append("CGRA simulator aggregate components have mismatched hardware")
                component_mapping_ids = {
                    component.get("mapping_id")
                    for component in components
                    if isinstance(component.get("mapping_id"), str)
                }
                declared_mapping_ids = set(aggregate_component_identities(data, "component_mapping_ids"))
                if declared_mapping_ids != component_mapping_ids:
                    diagnostics.append("CGRA simulator aggregate component_mapping_ids do not match components")
                for key in (
                    "dfg_cycles",
                    "hardware_aware_cycles",
                    "route_latency_cycles",
                    "memory_latency_cycles",
                    "temporal_penalty_cycles",
                    "performance_delta_cycles",
                    "route_segments",
                    "config_records",
                    "routed_edges",
                ):
                    value = data.get(key)
                    component_sum = sum(
                        int(component[key])
                        for component in components
                        if isinstance(component.get(key), int)
                    )
                    if isinstance(value, int) and value != component_sum:
                        diagnostics.append(f"CGRA simulator aggregate {key} does not match component sum")
    if kind == "sim_comparison_report":
        if data.get("kind") != "sim_comparison_report":
            diagnostics.append("simulation comparison report kind must be sim_comparison_report")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("simulation comparison report status must be a known status")
        elif data.get("status") != "pass":
            diagnostics.append("simulation comparison report status must be pass")
        for key in (
            "comparison_id",
            "workload",
            "runtime_input_identity",
            "dfg_sim_report_identity",
            "cgra_sim_report_identity",
        ):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"simulation comparison report lacks {key}")
        for key in (
            "functional_comparison_status",
            "memory_comparison_status",
            "performance_comparison_status",
        ):
            if data.get(key) not in BASE_STATUSES:
                diagnostics.append(f"simulation comparison report {key} must be a known status")
        allowed_classifications = {
            "match",
            "expected_hardware_constraint",
            "metric_not_comparable",
            "unsupported_scope",
            "mapping_invalid",
            "functional_mismatch",
            "report_mismatch",
        }
        if data.get("difference_classification") not in allowed_classifications:
            diagnostics.append("simulation comparison report has unknown difference_classification")
        definitions = data.get("performance_metric_definitions")
        if not isinstance(definitions, dict):
            diagnostics.append("simulation comparison report performance_metric_definitions must be an object")
        else:
            for key in ("dfg", "cgra"):
                if not isinstance(definitions.get(key), str) or not definitions.get(key):
                    diagnostics.append(f"simulation comparison report lacks {key} metric definition")
        explanation_categories = data.get("explanation_categories")
        if not isinstance(explanation_categories, list):
            diagnostics.append("simulation comparison report explanation_categories must be a list")
        diagnostics_list = data.get("diagnostics")
        if not isinstance(diagnostics_list, list):
            diagnostics.append("simulation comparison report diagnostics must be a list")
        dfg_cycles = data.get("dfg_sim_cycles")
        cgra_cycles = data.get("cgra_sim_cycles")
        if dfg_cycles is not None and (not isinstance(dfg_cycles, int) or dfg_cycles < 0):
            diagnostics.append("simulation comparison report dfg_sim_cycles must be non-negative integer or null")
        if cgra_cycles is not None and (not isinstance(cgra_cycles, int) or cgra_cycles < 0):
            diagnostics.append("simulation comparison report cgra_sim_cycles must be non-negative integer or null")
        if isinstance(dfg_cycles, int) and isinstance(cgra_cycles, int):
            if cgra_cycles < dfg_cycles and data.get("difference_classification") != "metric_not_comparable":
                diagnostics.append("simulation comparison report needs non-comparable classification for optimistic CGRA cycles")
            if (
                cgra_cycles > dfg_cycles
                and data.get("difference_classification") == "expected_hardware_constraint"
                and isinstance(explanation_categories, list)
                and not explanation_categories
            ):
                diagnostics.append("simulation comparison report needs explanation categories for hardware overhead")
        delta = data.get("performance_delta_cycles")
        if delta is not None and (not isinstance(delta, int) or delta < 0):
            diagnostics.append("simulation comparison report performance_delta_cycles must be non-negative integer or null")
        validate_sim_comparison_references(path, data, diagnostics)
    if kind == "pnr_mapping_artifact":
        if data.get("kind") != "pnr_mapping":
            diagnostics.append("PnR mapping artifact kind must be pnr_mapping")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("PnR mapping artifact status must be a known status")
        placements = data.get("placements")
        routes = data.get("routes")
        bitstream = data.get("config_bitstream")
        placed_records = data.get("placed_records")
        routed_edges = data.get("routed_edges")
        unrouted_edges = data.get("unrouted_edges")
        unplaced_records = data.get("unplaced_records")
        config_records = data.get("config_records")
        if not isinstance(placements, list):
            diagnostics.append("PnR mapping artifact placements must be a list")
        if not isinstance(routes, list):
            diagnostics.append("PnR mapping artifact routes must be a list")
        if not isinstance(bitstream, list):
            diagnostics.append("PnR mapping artifact config_bitstream must be a list")
        if not isinstance(placed_records, int) or placed_records < 0:
            diagnostics.append("PnR mapping artifact placed_records must be non-negative integer")
        if not isinstance(routed_edges, int) or routed_edges < 0:
            diagnostics.append("PnR mapping artifact routed_edges must be non-negative integer")
        if not isinstance(unrouted_edges, int) or unrouted_edges < 0:
            diagnostics.append("PnR mapping artifact unrouted_edges must be non-negative integer")
        if not isinstance(unplaced_records, int) or unplaced_records < 0:
            diagnostics.append("PnR mapping artifact unplaced_records must be non-negative integer")
        if not isinstance(config_records, int) or config_records < 0:
            diagnostics.append("PnR mapping artifact config_records must be non-negative integer")
        if isinstance(placements, list) and isinstance(placed_records, int) and placed_records != len(placements):
            diagnostics.append("PnR mapping artifact placed_records does not match placements size")
        if isinstance(routes, list) and isinstance(routed_edges, int) and routed_edges != len(routes):
            diagnostics.append("PnR mapping artifact routed_edges does not match routes size")
        if isinstance(routes, list):
            for index, route in enumerate(routes, start=1):
                if not isinstance(route, dict):
                    diagnostics.append(f"PnR mapping artifact route {index} must be an object")
                    continue
                for key in (
                    "record_id",
                    "edge_ref",
                    "producer_binding",
                    "consumer_binding",
                    "payload_kind",
                    "from",
                    "to",
                ):
                    if not isinstance(route.get(key), str) or not route.get(key):
                        diagnostics.append(f"PnR mapping artifact route {index} lacks {key}")
                segments = route.get("segments")
                if not isinstance(segments, list) or not segments:
                    diagnostics.append(f"PnR mapping artifact route {index} lacks non-empty segments")
                    continue
                for segment_index, segment in enumerate(segments, start=1):
                    if not isinstance(segment, dict):
                        diagnostics.append(
                            f"PnR mapping artifact route {index} segment {segment_index} must be an object"
                        )
                        continue
                    for key in ("segment_id", "segment_kind", "source_endpoint", "sink_endpoint"):
                        if not isinstance(segment.get(key), str) or not segment.get(key):
                            diagnostics.append(
                                f"PnR mapping artifact route {index} segment {segment_index} lacks {key}"
                            )
        if isinstance(bitstream, list) and isinstance(config_records, int) and config_records != len(bitstream):
            diagnostics.append("PnR mapping artifact config_records does not match config_bitstream size")
        if is_workload_graph_set_aggregate(data):
            components = aggregate_component_reports(
                path,
                data,
                "component_mapping_artifact_identities",
                "pnr_mapping_artifact",
                diagnostics,
                "PnR mapping aggregate artifact",
            )
            if components:
                workload = data.get("workload")
                hardware = data.get("hardware")
                if any(component.get("workload") != workload for component in components):
                    diagnostics.append("PnR mapping aggregate components have mismatched workload")
                if any(component.get("hardware") != hardware for component in components):
                    diagnostics.append("PnR mapping aggregate components have mismatched hardware")
                component_mapping_ids = {
                    component.get("mapping_id")
                    for component in components
                    if isinstance(component.get("mapping_id"), str)
                }
                declared_mapping_ids = set(aggregate_component_identities(data, "component_mapping_ids"))
                if declared_mapping_ids != component_mapping_ids:
                    diagnostics.append("PnR mapping aggregate component_mapping_ids do not match components")
                for key in (
                    "placed_records",
                    "routed_edges",
                    "unrouted_edges",
                    "unplaced_records",
                    "config_records",
                ):
                    value = data.get(key)
                    component_sum = sum(
                        int(component[key])
                        for component in components
                        if isinstance(component.get(key), int)
                    )
                    if isinstance(value, int) and value != component_sum:
                        diagnostics.append(f"PnR mapping aggregate {key} does not match component sum")
    if kind == "runtime_package":
        if data.get("kind") != "runtime_package":
            diagnostics.append("runtime package kind must be runtime_package")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("runtime package status must be a known status")
        for key in (
            "package_id",
            "workload",
            "work_package_identity",
            "launch_descriptor_identity",
            "host_program_identity",
            "host_wrapper_identity",
            "synchronization_mode",
        ):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"runtime package lacks {key}")
        for key in ("selected_mapping_artifact_identity", "fabric_adg_identity"):
            if not isinstance(data.get(key), str):
                diagnostics.append(f"runtime package {key} must be a string")
        validate_work_package_metadata(data.get("work_package_metadata"), data, diagnostics)
        target_profile = data.get("target_profile")
        if not isinstance(target_profile, dict):
            diagnostics.append("runtime package target_profile must be an object")
            target_profile = {}
        else:
            for key in ("target_kind", "profile_id"):
                if not isinstance(target_profile.get(key), str) or not target_profile.get(key):
                    diagnostics.append(f"runtime package target_profile lacks {key}")
            if target_profile.get("target_kind") == "simulator":
                if not isinstance(target_profile.get("simulator"), str) or not target_profile.get("simulator"):
                    diagnostics.append("runtime package simulator target lacks simulator")
                elif target_profile.get("simulator") not in {"cgra_sim", "dfg_sim", "rtl_sim"}:
                    diagnostics.append("runtime package has unsupported simulator target")
            elif target_profile.get("target_kind") == "hardware":
                if not isinstance(target_profile.get("hardware_backend"), str) or not target_profile.get("hardware_backend"):
                    diagnostics.append("runtime package hardware target lacks hardware_backend")
            elif target_profile.get("target_kind"):
                diagnostics.append("runtime package has unsupported target_kind")
        fallback_policy = data.get("fallback_policy")
        if fallback_policy not in {
            "require_acceleration",
            "allow_host_fallback",
            "allow_scalar_fallback",
            "report_only",
        }:
            diagnostics.append("runtime package has unknown fallback_policy")
        validate_fallback_decision(
            data.get("fallback_decision"),
            diagnostics,
            "runtime package",
            expected_policy=fallback_policy,
            target_profile_id=target_profile.get("profile_id"),
            require_complete=True,
        )
        validate_runtime_configuration(
            data.get("runtime_configuration"),
            data,
            target_profile,
            diagnostics,
        )
        validate_host_interface(data.get("host_interface"), data, diagnostics)
        validate_runtime_report(data.get("runtime_report"), data, diagnostics)
        validate_report_output_configuration(data.get("report_output_configuration"), data, diagnostics)
        data_movement_policy = data.get("data_movement_policy")
        if data_movement_policy not in DATA_MOVEMENT_POLICIES:
            diagnostics.append("runtime package has unknown data_movement_policy")
        diagnostics_list = data.get("diagnostics")
        if not isinstance(diagnostics_list, list):
            diagnostics.append("runtime package diagnostics must be a list")
        diagnostic_records = validate_runtime_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "runtime package",
            source_provenance=runtime_source_provenance(data),
            host_wrapper_identity=data.get("host_wrapper_identity"),
        )
        memory_descriptors = data.get("memory_descriptors")
        if not isinstance(memory_descriptors, list):
            diagnostics.append("runtime package memory_descriptors must be a list")
            memory_descriptors = []
        argument_descriptors = data.get("argument_descriptors")
        if not isinstance(argument_descriptors, list):
            diagnostics.append("runtime package argument_descriptors must be a list")
            argument_descriptors = []
        required_features = data.get("required_runtime_features")
        if not isinstance(required_features, list):
            diagnostics.append("runtime package required_runtime_features must be a list")
            required_features = []
        elif any(not isinstance(feature, str) or not feature for feature in required_features):
            diagnostics.append("runtime package required_runtime_features entries must be non-empty strings")
        required_data_movement_policies = data.get("required_data_movement_policies")
        if not isinstance(required_data_movement_policies, list):
            diagnostics.append("runtime package required_data_movement_policies must be a list")
            required_data_movement_policies = []
        elif any(
            not isinstance(policy, str) or not policy
            for policy in required_data_movement_policies
        ):
            diagnostics.append("runtime package required_data_movement_policies entries must be non-empty strings")
        else:
            for policy in required_data_movement_policies:
                if policy not in DATA_MOVEMENT_POLICIES:
                    diagnostics.append(f"runtime package required_data_movement_policies has unknown policy {policy}")
        required_synchronization_policies = data.get("required_synchronization_policies")
        if not isinstance(required_synchronization_policies, list):
            diagnostics.append("runtime package required_synchronization_policies must be a list")
            required_synchronization_policies = []
        elif any(
            not isinstance(policy, str) or not policy
            for policy in required_synchronization_policies
        ):
            diagnostics.append("runtime package required_synchronization_policies entries must be non-empty strings")
        else:
            for policy in required_synchronization_policies:
                if policy not in SYNCHRONIZATION_POLICIES:
                    diagnostics.append(
                        f"runtime package required_synchronization_policies has unknown policy {policy}"
                    )
        simulator_reports = data.get("simulator_report_identities")
        if not isinstance(simulator_reports, list):
            diagnostics.append("runtime package simulator_report_identities must be a list")
            simulator_reports = []
        elif any(not isinstance(identity, str) or not identity for identity in simulator_reports):
            diagnostics.append("runtime package simulator_report_identities entries must be non-empty strings")
        input_fingerprints = data.get("input_artifact_fingerprints")
        if not isinstance(input_fingerprints, dict):
            diagnostics.append("runtime package input_artifact_fingerprints must be an object")
            input_fingerprints = {}
        else:
            for identity, fingerprint in input_fingerprints.items():
                if not isinstance(identity, str) or not identity:
                    diagnostics.append("runtime package input_artifact_fingerprints has invalid identity")
                    continue
                if not valid_sha256_hex(fingerprint):
                    diagnostics.append(
                        f"runtime package input_artifact_fingerprints has invalid fingerprint for {identity}"
                    )
                    continue
                resolved = resolve_artifact_identity_reference(path, identity)
                if resolved is not None and fingerprint != artifact_fingerprint(resolved):
                    diagnostics.append(f"runtime package input_artifact_fingerprints stale for {identity!r}")
        runtime_configuration = data.get("runtime_configuration")
        runtime_custom_policy = None
        runtime_platform_binding = None
        if isinstance(runtime_configuration, dict):
            runtime_custom_policy = runtime_configuration.get("custom_data_movement_policy_identity")
            runtime_platform_binding = runtime_configuration.get("platform_binding_identity")
        for index, descriptor in enumerate(memory_descriptors, start=1):
            if not isinstance(descriptor, dict):
                diagnostics.append(f"runtime package memory descriptor {index} must be an object")
                continue
            for key in (
                "logical_argument",
                "host_buffer_identity",
                "direction",
                "policy",
                "runtime_input_identity",
                "element_layout",
                "address_space",
                "coherence_requirement",
                "transfer_policy",
            ):
                if not isinstance(descriptor.get(key), str) or not descriptor.get(key):
                    diagnostics.append(f"runtime package memory descriptor {index} lacks {key}")
            for key in ("byte_size", "alignment_bytes"):
                if not isinstance(descriptor.get(key), int) or descriptor.get(key) <= 0:
                    diagnostics.append(f"runtime package memory descriptor {index} has invalid {key}")
            descriptor_policy = descriptor.get("policy")
            if descriptor_policy not in DATA_MOVEMENT_POLICIES:
                diagnostics.append(f"runtime package memory descriptor {index} has unknown policy")
            if descriptor_policy != data_movement_policy:
                diagnostics.append(
                    f"runtime package memory descriptor {index} policy does not match data_movement_policy"
                )
            expected_coherence = COHERENCE_REQUIREMENT_BY_POLICY.get(data_movement_policy)
            if expected_coherence is not None and descriptor.get("coherence_requirement") != expected_coherence:
                diagnostics.append(
                    f"runtime package memory descriptor {index} coherence_requirement does not match policy"
                )
            if (
                data_movement_policy == "simulated"
                and descriptor.get("address_space") != SIMULATOR_MEMORY_ADDRESS_SPACE
            ):
                diagnostics.append(
                    f"runtime package memory descriptor {index} address_space does not match simulated policy"
                )
            if descriptor.get("transfer_policy") != data_movement_policy:
                diagnostics.append(
                    f"runtime package memory descriptor {index} transfer_policy does not match data_movement_policy"
                )
            platform_binding = descriptor.get("platform_binding_identity")
            if platform_binding is not None and (not isinstance(platform_binding, str) or not platform_binding):
                diagnostics.append(
                    f"runtime package memory descriptor {index} has invalid platform_binding_identity"
                )
            if isinstance(runtime_platform_binding, str) and runtime_platform_binding:
                if descriptor.get("platform_binding_identity") != runtime_platform_binding:
                    diagnostics.append(
                        f"runtime package memory descriptor {index} "
                        "platform_binding_identity does not match runtime configuration"
                    )
                expected_address_space = f"{runtime_platform_binding}::address_space"
                if descriptor.get("address_space") != expected_address_space:
                    diagnostics.append(
                        f"runtime package memory descriptor {index} "
                        "address_space does not match platform binding"
                    )
            descriptor_custom_policy = descriptor.get("custom_data_movement_policy_identity")
            if descriptor_policy == "custom":
                if not isinstance(descriptor_custom_policy, str) or not descriptor_custom_policy:
                    diagnostics.append(
                        f"runtime package memory descriptor {index} lacks custom_data_movement_policy_identity"
                    )
                elif isinstance(runtime_custom_policy, str) and descriptor_custom_policy != runtime_custom_policy:
                    diagnostics.append(
                        f"runtime package memory descriptor {index} custom policy does not match configuration"
                    )
            elif descriptor_custom_policy is not None:
                diagnostics.append(
                    f"runtime package memory descriptor {index} custom policy is only valid for custom policy"
                )
        for index, descriptor in enumerate(argument_descriptors, start=1):
            if not isinstance(descriptor, dict):
                diagnostics.append(f"runtime package argument descriptor {index} must be an object")
                continue
            for key in ("name", "identity", "descriptor_kind"):
                if not isinstance(descriptor.get(key), str) or not descriptor.get(key):
                    diagnostics.append(f"runtime package argument descriptor {index} lacks {key}")
            validate_runtime_argument_descriptor_kind(descriptor, diagnostics, "runtime package", index)
            validate_runtime_argument_descriptor_identity(
                descriptor,
                diagnostics,
                "runtime package",
                index,
                runtime_argument_identity_expectations(data),
            )
        artifact_input_references = runtime_artifact_input_references(
            data.get("selected_mapping_artifact_identity"),
            simulator_reports,
            argument_descriptors,
        )
        validate_runtime_package_sim_comparison_reference(path, data, diagnostics)
        validate_runtime_package_simulator_report_references(path, data, diagnostics)
        validate_runtime_package_mapping_reference(path, data, diagnostics)
        for identity in input_fingerprints:
            if identity not in artifact_input_references:
                diagnostics.append(
                    f"runtime package input_artifact_fingerprints references {identity!r} outside runtime inputs"
                )
        for reference in artifact_input_references:
            if reference not in input_fingerprints:
                diagnostics.append(f"runtime package input_artifact_fingerprints lacks {reference!r}")
        report_output_configuration = data.get("report_output_configuration")
        if isinstance(report_output_configuration, dict):
            if (
                report_output_configuration.get("trace_output_enabled") is True
                and "runtime_trace_output" not in required_features
            ):
                diagnostics.append("runtime package trace output requires runtime_trace_output feature")
            if (
                report_output_configuration.get("profiling_output_enabled") is True
                and "runtime_profiling_output" not in required_features
            ):
                diagnostics.append("runtime package profiling output requires runtime_profiling_output feature")
        if data_movement_policy not in required_data_movement_policies:
            diagnostics.append("runtime package required_data_movement_policies omits data_movement_policy")
        synchronization_mode = data.get("synchronization_mode")
        if synchronization_mode not in SYNCHRONIZATION_POLICIES:
            diagnostics.append("runtime package has unknown synchronization_mode")
        if synchronization_mode not in required_synchronization_policies:
            diagnostics.append("runtime package required_synchronization_policies omits synchronization_mode")
        validate_runtime_launch_descriptor(
            data.get("launch_descriptor"),
            data,
            target_profile,
            memory_descriptors,
            argument_descriptors,
            diagnostics,
        )
        validate_runtime_handle_model(data.get("runtime_handle_model"), diagnostics)
        if data.get("status") == "pass":
            if not memory_descriptors:
                diagnostics.append("runtime package pass needs memory_descriptors")
            if not argument_descriptors:
                diagnostics.append("runtime package pass needs argument_descriptors")
            if not required_features:
                diagnostics.append("runtime package pass needs required_runtime_features")
            if not required_data_movement_policies:
                diagnostics.append("runtime package pass needs required_data_movement_policies")
            if not required_synchronization_policies:
                diagnostics.append("runtime package pass needs required_synchronization_policies")
            simulator = target_profile.get("simulator")
            if simulator in {"cgra_sim", "dfg_sim"} and not simulator_reports:
                diagnostics.append("runtime package pass needs simulator_report_identities")
            if not input_fingerprints:
                diagnostics.append("runtime package pass needs input_artifact_fingerprints")
            if target_profile.get("target_kind") == "simulator" and data_movement_policy != "simulated":
                diagnostics.append("runtime package simulator target needs simulated data movement policy")
            argument_names = {
                descriptor.get("name")
                for descriptor in argument_descriptors
                if isinstance(descriptor, dict)
            }
            if simulator == "cgra_sim":
                if not data.get("selected_mapping_artifact_identity"):
                    diagnostics.append("runtime package CGRA-sim target needs mapping artifact identity")
                if not data.get("fabric_adg_identity"):
                    diagnostics.append("runtime package CGRA-sim target needs Fabric ADG identity")
                if "mapping_artifact" not in argument_names:
                    diagnostics.append("runtime package CGRA-sim target needs mapping artifact descriptor")
                if not any(str(identity).endswith("cgra-sim-report") for identity in simulator_reports):
                    diagnostics.append("runtime package CGRA-sim target needs CGRA-sim report identity")
                for identity in [data.get("selected_mapping_artifact_identity"), *simulator_reports]:
                    if isinstance(identity, str) and identity and identity not in input_fingerprints:
                        diagnostics.append(f"runtime package lacks input fingerprint for {identity}")
            if simulator == "dfg_sim":
                if data.get("selected_mapping_artifact_identity"):
                    diagnostics.append("runtime package DFG-sim target must not require mapping artifact identity")
                if data.get("fabric_adg_identity"):
                    diagnostics.append("runtime package DFG-sim target must not require Fabric ADG identity")
                if "mapping_artifact" in argument_names:
                    diagnostics.append("runtime package DFG-sim target must not include mapping artifact descriptor")
                if "dfg_sim_report" not in argument_names:
                    diagnostics.append("runtime package DFG-sim target needs DFG-sim report descriptor")
                if not any(str(identity).endswith("dfg-sim-report") for identity in simulator_reports):
                    diagnostics.append("runtime package DFG-sim target needs DFG-sim report identity")
                for identity in simulator_reports:
                    if isinstance(identity, str) and identity and identity not in input_fingerprints:
                        diagnostics.append(f"runtime package lacks input fingerprint for {identity}")
        elif not diagnostic_records:
            diagnostics.append("runtime package non-pass status needs diagnostic_records")
    if kind == "eda_report":
        if data.get("kind") != "eda_report":
            diagnostics.append("EDA report kind must be eda_report")
        if data.get("status") not in BASE_STATUSES:
            diagnostics.append("EDA report status must be a known status")
        for key in ("report_id", "rtl_manifest_identity", "tool_profile_id", "tool_name", "command_role"):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"EDA report lacks {key}")
        if data.get("capability_class") != "rtl_lint":
            diagnostics.append("EDA report capability_class must be rtl_lint")
        if data.get("command_role") != "rtl lint":
            diagnostics.append("EDA report command_role must be rtl lint")
        if not isinstance(data.get("tool_version"), str):
            diagnostics.append("EDA report tool_version must be a string")
        for key in ("checked_top_modules", "checked_source_files", "diagnostics"):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"EDA report {key} must be a list")
        diagnostic_records = validate_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "EDA report",
        )
        returncode = data.get("returncode")
        if returncode is not None and not isinstance(returncode, int):
            diagnostics.append("EDA report returncode must be an integer or null")
        if data.get("status") == "pass":
            if not data.get("tool_version"):
                diagnostics.append("EDA report pass needs tool_version")
            if returncode != 0:
                diagnostics.append("EDA report pass needs zero returncode")
            if diagnostic_records:
                diagnostics.append("EDA report pass must not carry diagnostic_records")
            if not data.get("checked_top_modules"):
                diagnostics.append("EDA report pass needs checked_top_modules")
            if not data.get("checked_source_files"):
                diagnostics.append("EDA report pass needs checked_source_files")
        elif data.get("status") == "fail":
            if not isinstance(returncode, int) or returncode == 0:
                diagnostics.append("EDA report fail needs non-zero returncode")
            if not diagnostic_records:
                diagnostics.append("EDA report fail needs diagnostic_records")
        elif data.get("status") in {"blocked", "unsupported", "not_run"} and not diagnostic_records:
            diagnostics.append("EDA report non-pass status needs diagnostic_records")
        validate_eda_report_input_fingerprints(path, data, diagnostics)
        validate_eda_report_source_fingerprints(path, data, diagnostics)
        validate_eda_report_rtl_manifest_reference(path, data, diagnostics)
    if kind == "workload_report_bundle":
        if data.get("kind") != "workload_report_bundle":
            diagnostics.append("workload report bundle kind must be workload_report_bundle")
        if data.get("report_status") not in BASE_STATUSES:
            diagnostics.append("workload report bundle report_status must be a known status")
        for key in (
            "bundle_id",
            "workload",
            "source_artifact_identity",
            "compiler_command_identity",
            "runtime_input_identity",
            "selected_hardware_candidate_identity",
            "selected_mapping_artifact_identity",
        ):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"workload report bundle lacks {key}")
        validate_fallback_decision(
            data.get("runtime_fallback_decision"),
            diagnostics,
            "workload report bundle runtime",
            require_complete=data.get("report_status") == "pass",
        )
        validate_workload_report_input_fingerprints(path, data, diagnostics)
        validate_workload_report_sim_comparison_reference(path, data, diagnostics)
        runtime_evidence = data.get("runtime_evidence")
        host_interface_expectations: dict[str, object] = {}
        if isinstance(runtime_evidence, dict):
            for key in ("host_program_identity", "host_wrapper_identity"):
                if key in runtime_evidence:
                    host_interface_expectations[key] = runtime_evidence.get(key)
        runtime_host_interface = data.get("runtime_host_interface")
        validate_host_interface(
            runtime_host_interface,
            host_interface_expectations,
            diagnostics,
            "workload report bundle runtime",
        )
        if isinstance(runtime_host_interface, dict):
            if runtime_host_interface.get("source_provenance") != data.get("runtime_input_identity"):
                diagnostics.append(
                    "workload report bundle runtime host_interface source_provenance does not match runtime input"
                )
        validate_workload_runtime_evidence_references(
            data,
            runtime_evidence,
            diagnostics,
        )
        validate_workload_runtime_package_projection(
            path,
            data,
            runtime_evidence,
            diagnostics,
        )
        validate_runtime_evidence(
            path,
            runtime_evidence,
            diagnostics,
            data.get("report_status") == "pass",
        )
        diagnostic_records = validate_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "workload report bundle",
        )
        if data.get("report_status") != "pass" and not diagnostic_records:
            diagnostics.append("workload report bundle non-pass status needs diagnostic_records")
        metrics = data.get("metric_records")
        metric_ids: set[str] = set()
        metric_class_by_id: dict[str, str] = {}
        metric_value_by_id: dict[str, float] = {}
        if not isinstance(metrics, list) or not metrics:
            diagnostics.append("workload report bundle needs non-empty metric_records")
            metrics = []
        for index, metric in enumerate(metrics, start=1):
            if not isinstance(metric, dict):
                diagnostics.append(f"workload report bundle metric {index} must be an object")
                continue
            metric_id = metric.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id:
                diagnostics.append(f"workload report bundle metric {index} lacks metric_id")
            elif metric_id in metric_ids:
                diagnostics.append(f"workload report bundle repeats metric_id {metric_id}")
            else:
                metric_ids.add(metric_id)
                metric_class = metric.get("metric_class")
                if isinstance(metric_class, str) and metric_class:
                    metric_class_by_id[metric_id] = metric_class
            for key in (
                "metric_class",
                "unit",
                "fidelity_level",
                "evidence_source_artifact_id",
                "producer_component",
                "derivation_kind",
            ):
                if not isinstance(metric.get(key), str) or not metric.get(key):
                    diagnostics.append(f"workload report bundle metric {index} lacks {key}")
            validate_report_metric_unit(metric, diagnostics, "workload report bundle", index)
            metric_class = metric.get("metric_class")
            derivation_kind = metric.get("derivation_kind")
            if isinstance(metric_class, str) and isinstance(derivation_kind, str):
                expected_derivation = EXPECTED_WORKLOAD_DERIVATION_BY_METRIC_CLASS.get(metric_class)
                if expected_derivation is not None and derivation_kind != expected_derivation:
                    diagnostics.append(
                        f"workload report bundle metric {index} derivation_kind does not match metric_class"
                    )
            value = metric.get("value")
            if not isinstance(value, (int, float)) or value < 0:
                diagnostics.append(f"workload report bundle metric {index} has invalid value")
            elif isinstance(metric_id, str) and metric_id:
                metric_value_by_id[metric_id] = float(value)
            metric_diagnostics = metric.get("diagnostics")
            if not isinstance(metric_diagnostics, list):
                diagnostics.append(f"workload report bundle metric {index} diagnostics must be a list")
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("metric_class") != "estimated_runtime":
                continue
            inputs = metric.get("input_metric_ids")
            if not isinstance(inputs, list) or not inputs:
                diagnostics.append("workload report bundle runtime metric lacks input_metric_ids")
                continue
            missing_inputs = [metric_id for metric_id in inputs if metric_id not in metric_ids]
            if missing_inputs:
                diagnostics.append(
                    f"workload report bundle runtime metric references missing inputs {missing_inputs}"
                )
            input_classes = {
                metric_class_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str) and metric_id in metric_class_by_id
            }
            has_cycle_source = bool({"hardware_cycles", "optimistic_steps"} & input_classes)
            if not has_cycle_source or "frequency" not in input_classes:
                diagnostics.append("workload report bundle runtime metric lacks cycle or frequency source inputs")
            input_value_by_class = {
                metric_class_by_id[metric_id]: metric_value_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str)
                and metric_id in metric_class_by_id
                and metric_id in metric_value_by_id
            }
            cycles = input_value_by_class.get("hardware_cycles")
            if cycles is None:
                cycles = input_value_by_class.get("optimistic_steps")
            frequency = input_value_by_class.get("frequency")
            runtime_value = metric.get("value")
            if (
                isinstance(runtime_value, (int, float))
                and cycles is not None
                and frequency is not None
                and frequency > 0
            ):
                expected_runtime = cycles / frequency
                if not nearly_equal(float(runtime_value), expected_runtime):
                    diagnostics.append("workload report bundle runtime metric value does not match inputs")
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("metric_class") != "energy":
                continue
            inputs = metric.get("input_metric_ids")
            if not isinstance(inputs, list) or not inputs:
                diagnostics.append("workload report bundle energy metric lacks input_metric_ids")
                continue
            missing_inputs = [metric_id for metric_id in inputs if metric_id not in metric_ids]
            if missing_inputs:
                diagnostics.append(
                    f"workload report bundle energy metric references missing inputs {missing_inputs}"
                )
            input_classes = {
                metric_class_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str) and metric_id in metric_class_by_id
            }
            has_runtime_source = (
                "estimated_runtime" in input_classes
                or {"hardware_cycles", "frequency"} <= input_classes
                or {"optimistic_steps", "frequency"} <= input_classes
            )
            if not has_runtime_source:
                diagnostics.append("workload report bundle energy metric lacks runtime source inputs")
            missing_power_inputs = sorted({"dynamic_power", "leakage_power"} - input_classes)
            if missing_power_inputs:
                diagnostics.append(
                    f"workload report bundle energy metric lacks power source inputs {missing_power_inputs}"
                )
            input_value_by_class = {
                metric_class_by_id[metric_id]: metric_value_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str)
                and metric_id in metric_class_by_id
                and metric_id in metric_value_by_id
            }
            frequency = input_value_by_class.get("frequency")
            cycles = input_value_by_class.get("hardware_cycles")
            if cycles is None:
                cycles = input_value_by_class.get("optimistic_steps")
            runtime = input_value_by_class.get("estimated_runtime")
            dynamic_power = input_value_by_class.get("dynamic_power")
            leakage_power = input_value_by_class.get("leakage_power")
            energy_value = metric.get("value")
            if (
                isinstance(energy_value, (int, float))
                and (runtime is not None or (cycles is not None and frequency is not None and frequency > 0))
                and dynamic_power is not None
                and leakage_power is not None
            ):
                runtime_us = runtime
                if runtime_us is None:
                    assert cycles is not None
                    assert frequency is not None
                    runtime_us = cycles / frequency
                expected_energy = (dynamic_power + leakage_power) * runtime_us
                if not nearly_equal(float(energy_value), expected_energy):
                    diagnostics.append("workload report bundle energy metric value does not match inputs")
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("metric_class") != "throughput":
                continue
            inputs = metric.get("input_metric_ids")
            if not isinstance(inputs, list) or not inputs:
                diagnostics.append("workload report bundle throughput metric lacks input_metric_ids")
                continue
            missing_inputs = [metric_id for metric_id in inputs if metric_id not in metric_ids]
            if missing_inputs:
                diagnostics.append(
                    f"workload report bundle throughput metric references missing inputs {missing_inputs}"
                )
            input_classes = {
                metric_class_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str) and metric_id in metric_class_by_id
            }
            missing_classes = sorted({"workload_size", "estimated_runtime"} - input_classes)
            if missing_classes:
                diagnostics.append(
                    f"workload report bundle throughput metric lacks source inputs {missing_classes}"
                )
            input_value_by_class = {
                metric_class_by_id[metric_id]: metric_value_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str)
                and metric_id in metric_class_by_id
                and metric_id in metric_value_by_id
            }
            workload_size = input_value_by_class.get("workload_size")
            runtime = input_value_by_class.get("estimated_runtime")
            throughput_value = metric.get("value")
            if (
                isinstance(throughput_value, (int, float))
                and workload_size is not None
                and runtime is not None
                and runtime > 0
            ):
                expected_throughput = workload_size / runtime * 1_000_000.0
                if not nearly_equal(float(throughput_value), expected_throughput):
                    diagnostics.append("workload report bundle throughput metric value does not match inputs")
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("metric_class") != "performance_per_watt":
                continue
            inputs = metric.get("input_metric_ids")
            if not isinstance(inputs, list) or not inputs:
                diagnostics.append("workload report bundle performance per watt metric lacks input_metric_ids")
                continue
            missing_inputs = [metric_id for metric_id in inputs if metric_id not in metric_ids]
            if missing_inputs:
                diagnostics.append(
                    f"workload report bundle performance per watt metric references missing inputs {missing_inputs}"
                )
            input_classes = {
                metric_class_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str) and metric_id in metric_class_by_id
            }
            missing_classes = sorted({"throughput", "dynamic_power", "leakage_power"} - input_classes)
            if missing_classes:
                diagnostics.append(
                    f"workload report bundle performance per watt metric lacks source inputs {missing_classes}"
                )
            input_value_by_class = {
                metric_class_by_id[metric_id]: metric_value_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str)
                and metric_id in metric_class_by_id
                and metric_id in metric_value_by_id
            }
            throughput = input_value_by_class.get("throughput")
            dynamic_power = input_value_by_class.get("dynamic_power")
            leakage_power = input_value_by_class.get("leakage_power")
            performance_value = metric.get("value")
            if (
                isinstance(performance_value, (int, float))
                and throughput is not None
                and dynamic_power is not None
                and leakage_power is not None
                and dynamic_power + leakage_power > 0
            ):
                expected_performance = throughput / ((dynamic_power + leakage_power) / 1000.0)
                if not nearly_equal(float(performance_value), expected_performance):
                    diagnostics.append(
                        "workload report bundle performance per watt metric value does not match inputs"
                    )
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("metric_class") != "performance_per_area":
                continue
            inputs = metric.get("input_metric_ids")
            if not isinstance(inputs, list) or not inputs:
                diagnostics.append("workload report bundle performance per area metric lacks input_metric_ids")
                continue
            missing_inputs = [metric_id for metric_id in inputs if metric_id not in metric_ids]
            if missing_inputs:
                diagnostics.append(
                    f"workload report bundle performance per area metric references missing inputs {missing_inputs}"
                )
            input_classes = {
                metric_class_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str) and metric_id in metric_class_by_id
            }
            missing_classes = sorted({"throughput", "area"} - input_classes)
            if missing_classes:
                diagnostics.append(
                    f"workload report bundle performance per area metric lacks source inputs {missing_classes}"
                )
            input_value_by_class = {
                metric_class_by_id[metric_id]: metric_value_by_id[metric_id]
                for metric_id in inputs
                if isinstance(metric_id, str)
                and metric_id in metric_class_by_id
                and metric_id in metric_value_by_id
            }
            throughput = input_value_by_class.get("throughput")
            area = input_value_by_class.get("area")
            performance_value = metric.get("value")
            if (
                isinstance(performance_value, (int, float))
                and throughput is not None
                and area is not None
                and area > 0
            ):
                expected_performance = throughput / area
                if not nearly_equal(float(performance_value), expected_performance):
                    diagnostics.append(
                        "workload report bundle performance per area metric value does not match inputs"
                    )
    if kind == "hardware_report_bundle":
        if data.get("kind") != "hardware_report_bundle":
            diagnostics.append("hardware report bundle kind must be hardware_report_bundle")
        if data.get("report_status") not in BASE_STATUSES:
            diagnostics.append("hardware report bundle report_status must be a known status")
        for key in ("bundle_id", "hardware_candidate_identity", "fabric_adg_identity"):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"hardware report bundle lacks {key}")
        for key in ("adg_builder_recipe_identity", "rtl_manifest_identity"):
            if not isinstance(data.get(key), str):
                diagnostics.append(f"hardware report bundle {key} must be a string")
        for key in (
            "eda_report_identities",
            "fpa_report_identities",
            "supported_workload_classes",
            "diagnostic_records",
            "diagnostics",
        ):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"hardware report bundle {key} must be a list")
        diagnostic_records = validate_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "hardware report bundle",
        )
        if data.get("report_status") != "pass" and not diagnostic_records:
            diagnostics.append("hardware report bundle non-pass status needs diagnostic_records")
        if data.get("report_status") == "pass":
            if not data.get("rtl_manifest_identity"):
                diagnostics.append("hardware report bundle pass needs RTL manifest identity")
            if not data.get("fpa_report_identities"):
                diagnostics.append("hardware report bundle pass needs FPA report identity")
            if not data.get("supported_workload_classes"):
                diagnostics.append("hardware report bundle pass needs supported workload classes")
        validate_hardware_report_input_fingerprints(path, data, diagnostics)
        validate_hardware_report_fpa_references(path, data, diagnostics)
        validate_hardware_report_adg_builder_recipe_reference(path, data, diagnostics)
        validate_hardware_report_rtl_manifest_reference(path, data, diagnostics)
        validate_hardware_report_eda_references(path, data, diagnostics)
        metrics = data.get("metric_records")
        metric_ids: set[str] = set()
        if not isinstance(metrics, list) or not metrics:
            diagnostics.append("hardware report bundle needs non-empty metric_records")
            metrics = []
        for index, metric in enumerate(metrics, start=1):
            if not isinstance(metric, dict):
                diagnostics.append(f"hardware report bundle metric {index} must be an object")
                continue
            metric_id = metric.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id:
                diagnostics.append(f"hardware report bundle metric {index} lacks metric_id")
            elif metric_id in metric_ids:
                diagnostics.append(f"hardware report bundle repeats metric_id {metric_id}")
            else:
                metric_ids.add(metric_id)
            for key in (
                "metric_class",
                "unit",
                "fidelity_level",
                "evidence_source_artifact_id",
                "producer_component",
                "derivation_kind",
            ):
                if not isinstance(metric.get(key), str) or not metric.get(key):
                    diagnostics.append(f"hardware report bundle metric {index} lacks {key}")
            validate_report_metric_unit(metric, diagnostics, "hardware report bundle", index)
            value = metric.get("value")
            if not isinstance(value, (int, float)) or value < 0:
                diagnostics.append(f"hardware report bundle metric {index} has invalid value")
            metric_diagnostics = metric.get("diagnostics")
            if not isinstance(metric_diagnostics, list):
                diagnostics.append(f"hardware report bundle metric {index} diagnostics must be a list")
    if kind == "dse_report_bundle":
        if data.get("kind") != "dse_report_bundle":
            diagnostics.append("DSE report bundle kind must be dse_report_bundle")
        if data.get("report_status") not in BASE_STATUSES:
            diagnostics.append("DSE report bundle report_status must be a known status")
        if not isinstance(data.get("dse_run_id"), str) or not data.get("dse_run_id"):
            diagnostics.append("DSE report bundle lacks dse_run_id")
        if data.get("report_status") == "pass":
            for key in ("selected_policy_id", "candidate_ordering_rule"):
                if not isinstance(data.get(key), str) or not data.get(key):
                    diagnostics.append(f"DSE report bundle lacks {key}")
            selected_policy_id = data.get("selected_policy_id")
            if isinstance(selected_policy_id, str) and data.get("dse_run_id") != f"dse::{selected_policy_id}":
                diagnostics.append("DSE report bundle dse_run_id does not match selected_policy_id")
        for key in (
            "objective_records",
            "candidate_list",
            "selected_candidates",
            "pareto_set",
            "rejected_candidate_summaries",
            "referenced_dse_candidate_artifact_identities",
            "referenced_workload_report_bundle_identities",
            "referenced_hardware_candidate_report_bundle_identities",
            "runtime_evidence_summaries",
            "diagnostic_records",
            "diagnostics",
        ):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"DSE report bundle {key} must be a list")
        diagnostic_records = validate_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "DSE report bundle",
        )
        if data.get("report_status") != "pass" and not diagnostic_records:
            diagnostics.append("DSE report bundle non-pass status needs diagnostic_records")
        validate_dse_report_input_fingerprints(path, data, diagnostics)
        validate_dse_report_candidate_references(path, data, diagnostics)
        validate_dse_report_workload_bundle_references(path, data, diagnostics)
        validate_dse_report_hardware_bundle_references(path, data, diagnostics)
        if not isinstance(data.get("policy_configuration"), dict):
            diagnostics.append("DSE report bundle policy_configuration must be an object")
            policy_configuration = {}
        else:
            policy_configuration = data["policy_configuration"]
        if data.get("report_status") == "pass" and (
            not isinstance(policy_configuration.get("conflict_resolution"), str)
            or not policy_configuration.get("conflict_resolution")
        ):
            diagnostics.append("DSE report bundle policy_configuration lacks conflict_resolution")
        if (
            data.get("report_status") == "pass"
            and isinstance(policy_configuration.get("conflict_resolution"), str)
            and policy_configuration.get("conflict_resolution") != "candidate_ordering_rule"
        ):
            diagnostics.append("DSE report bundle policy_configuration conflict_resolution does not match ordering rule")
        policy_kind = policy_configuration.get("policy_kind")
        if data.get("report_status") == "pass" and (
            not isinstance(policy_kind, str) or not policy_kind
        ):
            diagnostics.append("DSE report bundle policy_configuration lacks policy_kind")
        random_seed = policy_configuration.get("random_seed")
        if policy_kind in {"stochastic", "seeded"} and not isinstance(random_seed, int):
            diagnostics.append("DSE report bundle stochastic policy_configuration needs random_seed")
        referenced_metric_ids = dse_report_referenced_metric_ids(path, data)
        objectives = data.get("objective_records")
        if not isinstance(objectives, list) or (data.get("report_status") == "pass" and not objectives):
            diagnostics.append("DSE report bundle needs non-empty objective_records")
            objectives = []
        objective_kinds: set[str] = set()
        objective_metric_inputs_by_id: dict[str, set[str]] = {}
        for index, objective in enumerate(objectives, start=1):
            if not isinstance(objective, dict):
                diagnostics.append(f"DSE report bundle objective {index} must be an object")
                continue
            for key in (
                "objective_id",
                "objective_kind",
                "constraint_or_optimization_mode",
                "comparison_direction",
                "units",
            ):
                if not isinstance(objective.get(key), str) or not objective.get(key):
                    diagnostics.append(f"DSE report bundle objective {index} lacks {key}")
            objective_kind = objective.get("objective_kind")
            if isinstance(objective_kind, str) and objective_kind:
                objective_kinds.add(objective_kind)
                semantics = dse_objective_semantics(objective_kind)
                if semantics is not None:
                    expected_direction, expected_units = semantics
                    if objective.get("comparison_direction") != expected_direction:
                        diagnostics.append(
                            f"DSE report bundle objective {index} comparison_direction does not match objective_kind"
                        )
                    if objective.get("units") != expected_units:
                        diagnostics.append(f"DSE report bundle objective {index} units does not match objective_kind")
            objective_id = objective.get("objective_id")
            if isinstance(objective_id, str) and isinstance(objective_kind, str):
                if objective_id != f"objective::{objective_kind}":
                    diagnostics.append(f"DSE report bundle objective {index} objective_id does not match objective_kind")
            for key in ("metric_inputs", "validity_conditions"):
                if not isinstance(objective.get(key), list) or not objective.get(key):
                    diagnostics.append(f"DSE report bundle objective {index} lacks {key}")
            metric_inputs = objective.get("metric_inputs")
            if isinstance(metric_inputs, list):
                objective_id = objective.get("objective_id")
                if isinstance(objective_id, str) and objective_id:
                    objective_metric_inputs_by_id[objective_id] = {
                        metric_id
                        for metric_id in metric_inputs
                        if isinstance(metric_id, str) and metric_id
                    }
                for metric_id in metric_inputs:
                    if not isinstance(metric_id, str) or not metric_id:
                        diagnostics.append(f"DSE report bundle objective {index} metric_inputs has invalid entry")
                    elif referenced_metric_ids and metric_id not in referenced_metric_ids:
                        diagnostics.append(
                            f"DSE report bundle objective {index} metric_inputs references {metric_id!r}"
                        )
            priority = objective.get("priority")
            if not isinstance(priority, (int, float)) or priority <= 0:
                diagnostics.append(f"DSE report bundle objective {index} has invalid priority")
        if data.get("report_status") == "pass" and len(objective_kinds) == 1:
            objective_kind = next(iter(objective_kinds))
            if data.get("candidate_ordering_rule") != dse_ordering_rule_for_objective(objective_kind):
                diagnostics.append("DSE report bundle candidate_ordering_rule does not match objective")
            selected_policy_id = data.get("selected_policy_id")
            if isinstance(selected_policy_id, str):
                policy_objective = dse_objective_for_known_policy_id(selected_policy_id)
                if policy_objective is not None and policy_objective != objective_kind:
                    diagnostics.append("DSE report bundle selected_policy_id does not match objective")
        objective_ids = {
            objective.get("objective_id")
            for objective in objectives
            if isinstance(objective, dict)
            and isinstance(objective.get("objective_id"), str)
            and objective.get("objective_id")
        }
        candidates = data.get("candidate_list")
        candidate_ids: set[str] = set()
        candidate_status_by_id: dict[str, object] = {}
        selected_record_ids: set[str] = set()
        pareto_record_ids: set[str] = set()
        rejected_candidate_ids: set[str] = set()
        if not isinstance(candidates, list) or (data.get("report_status") == "pass" and not candidates):
            diagnostics.append("DSE report bundle needs non-empty candidate_list")
            candidates = []
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                diagnostics.append(f"DSE report bundle candidate {index} must be an object")
                continue
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                diagnostics.append(f"DSE report bundle candidate {index} lacks candidate_id")
            elif candidate_id in candidate_ids:
                diagnostics.append(f"DSE report bundle repeats candidate_id {candidate_id}")
            else:
                candidate_ids.add(candidate_id)
                candidate_status_by_id[candidate_id] = candidate.get("status")
            for key in ("candidate_kind", "status"):
                if not isinstance(candidate.get(key), str) or not candidate.get(key):
                    diagnostics.append(f"DSE report bundle candidate {index} lacks {key}")
            if candidate.get("status") not in SELECTION_STATUSES:
                diagnostics.append(f"DSE report bundle candidate {index} has unknown status")
            elif isinstance(candidate_id, str) and candidate_id:
                if candidate.get("status") == "selected":
                    selected_record_ids.add(candidate_id)
                elif candidate.get("status") == "pareto":
                    pareto_record_ids.add(candidate_id)
                elif candidate.get("status") == "rejected":
                    rejected_candidate_ids.add(candidate_id)
            for key in (
                "parent_candidate_ids",
                "referenced_input_artifacts",
                "generated_output_artifacts",
                "objective_records_used",
                "metric_records_used",
                "diagnostics",
            ):
                if not isinstance(candidate.get(key), list):
                    diagnostics.append(f"DSE report bundle candidate {index} {key} must be a list")
            objective_records_used = candidate.get("objective_records_used")
            if isinstance(objective_records_used, list):
                if candidate.get("status") in {"selected", "pareto"} and not objective_records_used:
                    diagnostics.append(
                        f"DSE report bundle candidate {index} needs objective_records_used"
                    )
                for objective_id in objective_records_used:
                    if not isinstance(objective_id, str) or not objective_id:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} objective_records_used has invalid entry"
                        )
                    elif objective_ids and objective_id not in objective_ids:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} objective_records_used references {objective_id!r}"
                        )
            metric_records_used = candidate.get("metric_records_used")
            if isinstance(metric_records_used, list):
                if candidate.get("status") in {"selected", "pareto"} and not metric_records_used:
                    diagnostics.append(
                        f"DSE report bundle candidate {index} needs metric_records_used"
                    )
                for metric_id in metric_records_used:
                    if not isinstance(metric_id, str) or not metric_id:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} metric_records_used has invalid entry"
                        )
                    elif referenced_metric_ids and metric_id not in referenced_metric_ids:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} metric_records_used references {metric_id!r}"
                        )
                if candidate.get("status") in {"selected", "pareto"} and isinstance(objective_records_used, list):
                    used_metrics = {
                        metric_id
                        for metric_id in metric_records_used
                        if isinstance(metric_id, str) and metric_id
                    }
                    for objective_id in objective_records_used:
                        if not isinstance(objective_id, str):
                            continue
                        missing_objective_metrics = sorted(
                            objective_metric_inputs_by_id.get(objective_id, set()) - used_metrics
                        )
                        if missing_objective_metrics:
                            diagnostics.append(
                                f"DSE report bundle candidate {index} lacks objective metric inputs "
                                f"{missing_objective_metrics}"
                            )
            generated_output_artifacts = candidate.get("generated_output_artifacts")
            if isinstance(generated_output_artifacts, list):
                if candidate.get("status") in {"selected", "pareto"} and not generated_output_artifacts:
                    diagnostics.append(
                        f"DSE report bundle candidate {index} needs generated_output_artifacts"
                    )
                for artifact in generated_output_artifacts:
                    if not isinstance(artifact, str) or not artifact:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} generated_output_artifacts has invalid entry"
                        )
                    elif resolve_artifact_identity_reference(path, artifact) is None:
                        diagnostics.append(
                            f"DSE report bundle candidate {index} generated_output_artifacts reference {artifact!r} does not exist"
                        )
            validate_dse_report_candidate_input_fingerprints(path, candidate, diagnostics, index)
        selected_candidates = data.get("selected_candidates")
        selected_candidate_ids = validate_dse_candidate_id_list(
            selected_candidates,
            diagnostics,
            "DSE report bundle selected_candidates",
        )
        if isinstance(selected_candidates, list):
            missing_selected = [
                candidate_id
                for candidate_id in selected_candidates
                if candidate_id not in candidate_ids
            ]
            if missing_selected:
                diagnostics.append(f"DSE report bundle selected candidates are missing records {missing_selected}")
            mismatched_selected = [
                candidate_id
                for candidate_id in selected_candidates
                if candidate_status_by_id.get(candidate_id) != "selected"
            ]
            if mismatched_selected:
                diagnostics.append(
                    f"DSE report bundle selected candidates have non-selected records {mismatched_selected}"
                )
            unlisted_selected = sorted(selected_record_ids - selected_candidate_ids)
            if unlisted_selected:
                diagnostics.append(
                    f"DSE report bundle selected candidate records are missing from selected_candidates {unlisted_selected}"
                )
        pareto_set = data.get("pareto_set")
        pareto_candidate_ids = validate_dse_candidate_id_list(
            pareto_set,
            diagnostics,
            "DSE report bundle pareto_set",
        )
        if isinstance(pareto_set, list):
            missing_pareto = [
                candidate_id
                for candidate_id in pareto_set
                if candidate_id not in candidate_ids
            ]
            if missing_pareto:
                diagnostics.append(f"DSE report bundle pareto candidates are missing records {missing_pareto}")
            mismatched_pareto = [
                candidate_id
                for candidate_id in pareto_set
                if candidate_status_by_id.get(candidate_id) != "pareto"
            ]
            if mismatched_pareto:
                diagnostics.append(
                    f"DSE report bundle pareto candidates have non-pareto records {mismatched_pareto}"
                )
            unlisted_pareto = sorted(pareto_record_ids - pareto_candidate_ids)
            if unlisted_pareto:
                diagnostics.append(
                    f"DSE report bundle Pareto candidate records are missing from pareto_set {unlisted_pareto}"
                )
        overlapping_selection = sorted(selected_candidate_ids & pareto_candidate_ids)
        if overlapping_selection:
            diagnostics.append(
                f"DSE report bundle selected and Pareto candidates overlap {overlapping_selection}"
            )
        rejected_summaries = data.get("rejected_candidate_summaries")
        if isinstance(rejected_summaries, list):
            rejected_summary_ids: set[str] = set()
            for index, summary in enumerate(rejected_summaries, start=1):
                if not isinstance(summary, dict):
                    diagnostics.append(f"DSE report bundle rejected summary {index} must be an object")
                    continue
                candidate_id = summary.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    diagnostics.append(f"DSE report bundle rejected summary {index} lacks candidate_id")
                else:
                    rejected_summary_ids.add(candidate_id)
                    if candidate_status_by_id.get(candidate_id) != "rejected":
                        diagnostics.append(
                            f"DSE report bundle rejected summary {index} does not reference a rejected candidate"
                        )
                summary_diagnostics = summary.get("diagnostics")
                if (
                    not isinstance(summary_diagnostics, list)
                    or any(not isinstance(item, str) for item in summary_diagnostics)
                ):
                    diagnostics.append(f"DSE report bundle rejected summary {index} diagnostics must be a string list")
            missing_rejected_summaries = sorted(rejected_candidate_ids - rejected_summary_ids)
            if missing_rejected_summaries:
                diagnostics.append(
                    "DSE report bundle rejected candidates are missing summaries "
                    f"{missing_rejected_summaries}"
                )
        if data.get("report_status") == "pass":
            for key in (
                "selected_candidates",
                "referenced_dse_candidate_artifact_identities",
                "referenced_workload_report_bundle_identities",
                "referenced_hardware_candidate_report_bundle_identities",
            ):
                if not data.get(key):
                    diagnostics.append(f"DSE report bundle pass needs {key}")
        validate_runtime_evidence_summaries(
            path,
            data.get("runtime_evidence_summaries"),
            diagnostics,
            data.get("report_status") == "pass",
            {
                identity
                for identity in data.get("referenced_workload_report_bundle_identities", [])
                if isinstance(identity, str) and identity
            },
        )
    entries_checked = len(data) if isinstance(data, dict) else 0
    if isinstance(data, dict) and kind == "artifact_manifest":
        entries_checked = manifest_entries_checked if manifest_entries_checked is not None else 0
    if isinstance(data, dict) and kind == "artifact_audit":
        reviews = data.get("artifact_reviews")
        entries_checked = len(reviews) if isinstance(reviews, list) else 0
    return {
        "artifact": str(path),
        "schema": kind,
        "entries_checked": entries_checked,
        "parser_checks": ["json_parse", "required_keys"],
        "finding": "pass" if not diagnostics else "fail",
        "diagnostics": diagnostics,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_by_kind(paths: Iterable[Path]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        schema = schema_for_path(path)
        if schema is None or not path.is_file():
            continue
        for row in read_csv_rows(path):
            copied = dict(row)
            copied["__path"] = str(path)
            grouped.setdefault(schema.kind, []).append(copied)
    return grouped


def json_objects_by_kind(paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        kind = json_kind_for_path(path)
        if kind is None or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data["__path"] = str(path)
            grouped.setdefault(kind, []).append(data)
    return grouped


def cross_finding(rule: str, message: str, row: dict[str, str]) -> dict[str, object]:
    return {
        "finding": "fail",
        "rule": rule,
        "message": message,
        "row_identity": {
            key: value
            for key, value in row.items()
            if key in {"candidate", "case", "hardware", "import_state", "kernel", "manifest_case", "workload"}
        },
    }


def valid_identity(value: str | None) -> bool:
    return value not in IGNORED_IDENTITIES


def valid_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def artifact_path(record: dict[str, object]) -> Path | None:
    path_text = record.get("__path")
    if not isinstance(path_text, str) or path_text == "":
        return None
    return Path(path_text).resolve()


def referenced_artifact_paths(row: dict[str, str], column: str) -> set[Path]:
    anchor_text = row.get("__path", "")
    anchor = Path(anchor_text) if anchor_text else Path(".")
    return {
        resolve_artifact_reference(anchor, reference)
        for reference in row.get(column, "").split(";")
        if reference
    }


def build_pass_mapping_artifacts_by_workload(
    json_grouped: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for artifact in json_grouped.get("pnr_mapping_artifact", []):
        workload = artifact.get("workload")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and artifact.get("status") == "pass"
            and isinstance(artifact.get("mapping_id"), str)
            and isinstance(artifact.get("hardware"), str)
        ):
            grouped.setdefault(workload, []).append(artifact)
    return grouped


def build_pass_dfg_reports_by_workload(
    json_grouped: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for report in json_grouped.get("dfg_sim_report", []):
        workload = report.get("workload")
        cycles = report.get("optimistic_cycles")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and report.get("status") == "pass"
            and isinstance(cycles, int)
            and cycles >= 0
        ):
            grouped.setdefault(workload, []).append(report)
    return prefer_workload_graph_set_aggregates(grouped)


def build_pass_cgra_reports_by_workload(
    json_grouped: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for report in json_grouped.get("cgra_sim_report", []):
        workload = report.get("workload")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and report.get("status") == "pass"
        ):
            grouped.setdefault(workload, []).append(report)
    return prefer_workload_graph_set_aggregates(grouped)


def route_segment_count(routes: object) -> int | None:
    if not isinstance(routes, list):
        return None
    count = 0
    for route in routes:
        if isinstance(route, dict) and isinstance(route.get("segments"), list):
            count += len(route["segments"])
    return count


def float_cell(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def nearly_equal(lhs: float, rhs: float, *, tolerance: float = 0.001) -> bool:
    return abs(lhs - rhs) <= tolerance


def cross_artifact_findings(paths: Iterable[Path]) -> list[dict[str, object]]:
    path_list = list(paths)
    grouped = rows_by_kind(path_list)
    json_grouped = json_objects_by_kind(path_list)
    findings: list[dict[str, object]] = []

    workloads = {
        row["workload"]
        for row in grouped.get("dataflow_primitive_coverage", [])
        if valid_identity(row.get("workload"))
    }
    hardware = {
        row["hardware"]
        for row in grouped.get("adg_hardware", [])
        if row.get("verify_status") == "pass" and valid_identity(row.get("hardware"))
    }
    hardware_symbol_counts: dict[str, int] = {}
    for candidate in hardware:
        symbol = candidate.rsplit("::", 1)[-1]
        hardware_symbol_counts[symbol] = hardware_symbol_counts.get(symbol, 0) + 1

    def canonical_hardware_ref(candidate: str | None) -> str | None:
        if not valid_identity(candidate):
            return None
        assert candidate is not None
        if not hardware:
            return candidate
        if candidate in hardware:
            return candidate
        if hardware_symbol_counts.get(candidate, 0) == 1:
            for full_ref in hardware:
                if full_ref.rsplit("::", 1)[-1] == candidate:
                    return full_ref
        return None

    pnr_rows = [
        row
        for row in grouped.get("pnr_mapping", [])
        if valid_identity(row.get("workload")) and valid_identity(row.get("hardware"))
    ]
    pnr_pairs = {(row["workload"], row["hardware"]) for row in pnr_rows}
    pass_mapping_artifacts_by_workload = build_pass_mapping_artifacts_by_workload(json_grouped)
    dfg_report_cycles_by_workload: dict[str, list[int]] = {}
    dfg_report_semantics_by_workload: dict[str, set[str]] = {}
    dfg_reports_by_workload_graph: dict[tuple[str, str], list[dict[str, object]]] = {}
    for workload, reports in build_pass_dfg_reports_by_workload(json_grouped).items():
        for report in reports:
            graph = report.get("graph")
            cycles = report.get("optimistic_cycles")
            semantics = report.get("operation_semantics_source")
            assert isinstance(cycles, int)
            dfg_report_cycles_by_workload.setdefault(workload, []).append(cycles)
            if isinstance(semantics, str):
                dfg_report_semantics_by_workload.setdefault(workload, set()).add(semantics)
            if isinstance(graph, str) and valid_identity(graph):
                dfg_reports_by_workload_graph.setdefault((workload, graph), []).append(report)
    for (workload, graph), reports in sorted(dfg_reports_by_workload_graph.items()):
        scale_points: list[tuple[int, int]] = []
        for report in reports:
            dynamic_work_items = report.get("dynamic_work_items")
            cycles = report.get("optimistic_cycles")
            if isinstance(dynamic_work_items, int) and isinstance(cycles, int):
                scale_points.append((dynamic_work_items, cycles))
        distinct_extents = sorted({extent for extent, _ in scale_points})
        if len(distinct_extents) <= 1:
            continue
        best_cycle_by_extent: dict[int, int] = {}
        for extent, cycles in scale_points:
            if extent not in best_cycle_by_extent or cycles < best_cycle_by_extent[extent]:
                best_cycle_by_extent[extent] = cycles
        previous_extent: int | None = None
        previous_cycles: int | None = None
        for extent in sorted(best_cycle_by_extent):
            cycles = best_cycle_by_extent[extent]
            if (
                previous_extent is not None
                and previous_cycles is not None
                and cycles <= previous_cycles
            ):
                findings.append(
                    cross_finding(
                        "dfg_cycle_monotonic_with_dynamic_work_items",
                        (
                            f"DFG-sim workload {workload!r} graph {graph!r} has "
                            f"{cycles} cycles at dynamic_work_items={extent}, "
                            f"not greater than {previous_cycles} cycles at "
                            f"dynamic_work_items={previous_extent}"
                        ),
                        {"workload": workload},
                    )
                )
            previous_extent = extent
            previous_cycles = cycles
    cgra_reports_by_workload = build_pass_cgra_reports_by_workload(json_grouped)

    def matching_pass_pnr_rows(
        workload: str,
        hardware_ref: str,
        mapping_id: str,
    ) -> list[dict[str, str]]:
        canonical_target = canonical_hardware_ref(hardware_ref)
        if canonical_target is None:
            return []
        matches: list[dict[str, str]] = []
        for candidate in pnr_rows:
            if candidate.get("status") != "pass":
                continue
            if candidate.get("workload") != workload:
                continue
            if candidate.get("mapping_id") != mapping_id:
                continue
            if canonical_hardware_ref(candidate.get("hardware")) == canonical_target:
                matches.append(candidate)
        return matches

    def matching_pass_sim_rows(workload: str) -> list[dict[str, str]]:
        return [
            row
            for row in grouped.get("sim_cycle", [])
            if row.get("status") == "pass" and row.get("kernel") == workload
        ]

    def matching_pass_fpa_rows(
        workload: str,
        hardware_ref: str,
    ) -> list[dict[str, str]]:
        canonical_target = canonical_hardware_ref(hardware_ref)
        if canonical_target is None:
            return []
        return [
            row
            for row in grouped.get("rtl_fpa", [])
            if row.get("status") == "pass"
            and row.get("workload") == workload
            and canonical_hardware_ref(row.get("hardware")) == canonical_target
        ]

    def matching_pass_mapping_artifacts(
        workload: str,
        hardware_ref: str,
        mapping_id: str,
    ) -> list[dict[str, object]]:
        canonical_target = canonical_hardware_ref(hardware_ref)
        if canonical_target is None:
            return []
        matches: list[dict[str, object]] = []
        for artifact in pass_mapping_artifacts_by_workload.get(workload, []):
            artifact_hardware = artifact.get("hardware")
            if not isinstance(artifact_hardware, str):
                continue
            if artifact.get("mapping_id") != mapping_id:
                continue
            if canonical_hardware_ref(artifact_hardware) == canonical_target:
                matches.append(artifact)
        return matches

    def matching_pass_cgra_reports(
        workload: str,
        hardware_ref: str,
        mapping_id: str,
    ) -> list[dict[str, object]]:
        canonical_target = canonical_hardware_ref(hardware_ref)
        if canonical_target is None:
            return []
        matches: list[dict[str, object]] = []
        for report in cgra_reports_by_workload.get(workload, []):
            report_hardware = report.get("hardware")
            if not isinstance(report_hardware, str):
                continue
            if report.get("mapping_id") != mapping_id:
                continue
            if canonical_hardware_ref(report_hardware) == canonical_target:
                matches.append(report)
        return matches

    inventory_rows = grouped.get("old_app_corpus_inventory", [])
    import_rows = grouped.get("app_import_status", [])
    if inventory_rows or import_rows:
        inventory_by_case = {row.get("case", ""): row for row in inventory_rows if valid_identity(row.get("case"))}
        imports_by_case: dict[str, list[dict[str, str]]] = {}
        for row in import_rows:
            case = row.get("case", "")
            if valid_identity(case):
                imports_by_case.setdefault(case, []).append(row)

        for case, rows in imports_by_case.items():
            if len(rows) > 1:
                findings.append(
                    cross_finding(
                        "app_import_unique_case",
                        f"app import status has duplicate rows for {case!r}",
                        rows[0],
                    )
                )

        for case in sorted(set(inventory_by_case) - set(imports_by_case)):
            findings.append(
                cross_finding(
                    "app_import_covers_inventory",
                    f"inventory case {case!r} has no import status row",
                    inventory_by_case[case],
                )
            )

        for row in import_rows:
            case = row.get("case", "")
            inventory = inventory_by_case.get(case)
            if inventory is None:
                findings.append(
                    cross_finding(
                        "app_import_case_resolves",
                        f"app import case {case!r} is absent from old app corpus inventory",
                        row,
                    )
                )
                continue
            state = row.get("import_state", "")
            manifest_case = row.get("manifest_case", "")
            inventory_status = inventory.get("status", "")
            if state == "accepted":
                if manifest_case != case:
                    findings.append(
                        cross_finding(
                            "app_import_accepted_manifest_case",
                            f"accepted app import case {case!r} must name the matching manifest case",
                            row,
                        )
                    )
                if inventory_status != "ready":
                    findings.append(
                        cross_finding(
                            "app_import_accepted_requires_ready_inventory",
                            f"accepted app import case {case!r} has inventory status {inventory_status!r}",
                            row,
                        )
                    )
            elif state == "deferred":
                if manifest_case:
                    findings.append(
                        cross_finding(
                            "app_import_deferred_has_no_manifest_case",
                            f"deferred app import case {case!r} must not name a manifest case",
                            row,
                        )
                    )
                if inventory_status != "ready":
                    findings.append(
                        cross_finding(
                            "app_import_deferred_requires_ready_inventory",
                            f"deferred app import case {case!r} has inventory status {inventory_status!r}",
                            row,
                        )
                    )
            elif state == "excluded":
                if inventory_status != "blocked":
                    findings.append(
                        cross_finding(
                            "app_import_excluded_requires_blocked_inventory",
                            f"excluded app import case {case!r} has inventory status {inventory_status!r}",
                            row,
                        )
                    )
                if not row.get("reason", ""):
                    findings.append(
                        cross_finding(
                            "app_import_excluded_has_reason",
                            f"excluded app import case {case!r} has no reason",
                            row,
                        )
                    )

    if workloads:
        for row in pnr_rows:
            if row["workload"] not in workloads:
                findings.append(
                    cross_finding(
                        "pnr_workload_resolves",
                        f"PnR workload {row['workload']!r} is absent from dataflow primitive coverage",
                        row,
                    )
                )
    for row in grouped.get("sim_cycle", []):
        kernel = row.get("kernel")
        if not valid_identity(kernel):
            continue
        if workloads and kernel not in workloads:
            findings.append(
                cross_finding(
                    "sim_workload_resolves",
                    f"sim kernel {kernel!r} is absent from dataflow primitive coverage",
                    row,
                )
            )
        if row.get("dfg_sim_cycles", ""):
            dfg_cycles = nonnegative_int_cell(row, "dfg_sim_cycles")
            report_cycles = dfg_report_cycles_by_workload.get(kernel, [])
            if report_cycles:
                report_cycle_total = sum(report_cycles)
                if dfg_cycles is not None and dfg_cycles != report_cycle_total:
                    findings.append(
                        cross_finding(
                            "sim_dfg_cycle_matches_dfg_report",
                            (
                                f"sim kernel {kernel!r} DFG-sim cycles "
                                f"{dfg_cycles} do not match summed DFG report cycles {report_cycle_total}"
                            ),
                            row,
                        )
                    )
            else:
                findings.append(
                    cross_finding(
                        "sim_dfg_cycle_requires_dfg_report",
                        (
                            f"sim kernel {kernel!r} has DFG-sim cycles "
                            "but no matching DFG-sim report artifact was provided"
                        ),
                        row,
                    )
                )
        if row.get("cgra_sim_cycles", ""):
            if not row.get("dfg_sim_cycles", ""):
                findings.append(
                    cross_finding(
                        "sim_cgra_cycle_requires_dfg_cycle",
                        f"sim kernel {kernel!r} has CGRA-sim cycles without comparable DFG-sim cycles",
                        row,
                    )
                )
            cgra_cycles = nonnegative_int_cell(row, "cgra_sim_cycles")
            cgra_reports = cgra_reports_by_workload.get(kernel, [])
            if not cgra_reports:
                findings.append(
                    cross_finding(
                        "sim_cgra_cycle_requires_cgra_report",
                        (
                            f"sim kernel {kernel!r} has CGRA-sim cycles "
                            "but no CGRA-sim report evidence was provided"
                        ),
                        row,
                    )
                )
            elif cgra_cycles is not None:
                reports_with_cycles = [
                    report
                    for report in cgra_reports
                    if isinstance(report.get("hardware_aware_cycles"), int)
                ]
                report_cycle_total = sum(
                    int(report["hardware_aware_cycles"]) for report in reports_with_cycles
                )
                matching_reports = reports_with_cycles if report_cycle_total == cgra_cycles else []
                if not matching_reports:
                    findings.append(
                        cross_finding(
                            "sim_cgra_cycle_matches_cgra_report",
                            (
                                f"sim kernel {kernel!r} CGRA-sim cycles "
                                f"{cgra_cycles} do not match summed CGRA report cycles {report_cycle_total}"
                            ),
                            row,
                        )
                    )
                else:
                    pass_mapping_artifacts = pass_mapping_artifacts_by_workload.get(kernel, [])
                    matching_mapping_reports = []
                    for report in matching_reports:
                        report_mapping_id = report.get("mapping_id")
                        report_hardware = canonical_hardware_ref(
                            report.get("hardware") if isinstance(report.get("hardware"), str) else None
                        )
                        if not isinstance(report_mapping_id, str) or report_hardware is None:
                            continue
                        for mapping in pass_mapping_artifacts:
                            mapping_hardware = canonical_hardware_ref(
                                mapping.get("hardware") if isinstance(mapping.get("hardware"), str) else None
                            )
                            if (
                                mapping.get("mapping_id") == report_mapping_id
                                and mapping_hardware == report_hardware
                            ):
                                route_segments = report.get("route_segments")
                                if isinstance(route_segments, int):
                                    mapped_segments = route_segment_count(mapping.get("routes"))
                                    if (
                                        mapped_segments is not None
                                        and mapped_segments != route_segments
                                    ):
                                        continue
                                config_records = report.get("config_records")
                                if (
                                    isinstance(config_records, int)
                                    and mapping.get("config_records") != config_records
                                ):
                                    continue
                                matching_mapping_reports.append(report)
                                break
                    if len(matching_mapping_reports) != len(matching_reports):
                        findings.append(
                            cross_finding(
                                "sim_cgra_report_matches_mapping",
                                (
                                    f"sim kernel {kernel!r} CGRA report does not "
                                    "match a pass PnR mapping artifact by hardware and mapping_id"
                                ),
                                row,
                            )
                        )
                    else:
                        matching_reports = matching_mapping_reports
                if matching_reports and row.get("dfg_sim_cycles", ""):
                    dfg_cycles = nonnegative_int_cell(row, "dfg_sim_cycles")
                    report_dfg_cycles = [
                        int(report["dfg_cycles"])
                        for report in matching_reports
                        if isinstance(report.get("dfg_cycles"), int)
                    ]
                    if dfg_cycles is not None and sum(report_dfg_cycles) != dfg_cycles:
                        findings.append(
                            cross_finding(
                                "sim_cgra_report_matches_dfg_cycle",
                                (
                                    f"sim kernel {kernel!r} CGRA report DFG cycles "
                                    "do not match summary DFG cycles"
                                ),
                                row,
                            )
                        )
                    dfg_semantics = dfg_report_semantics_by_workload.get(kernel, set())
                    if dfg_semantics and not all(
                        report.get("operation_semantics_source") in dfg_semantics
                        for report in matching_reports
                    ):
                        findings.append(
                            cross_finding(
                                "sim_cgra_report_uses_dfg_operation_semantics",
                                (
                                    f"sim kernel {kernel!r} CGRA report does not "
                                    "use the DFG-sim operation semantics source"
                                ),
                                row,
                            )
                        )
            pass_mapping_artifacts = pass_mapping_artifacts_by_workload.get(kernel, [])
            if not pass_mapping_artifacts:
                findings.append(
                    cross_finding(
                        "sim_cgra_cycle_requires_mapping",
                        (
                            f"sim kernel {kernel!r} has CGRA-sim cycles "
                            "but no pass PnR mapping artifact with mapping_id was provided"
                        ),
                        row,
                    )
                )
            elif hardware and not any(
                canonical_hardware_ref(
                    mapping.get("hardware") if isinstance(mapping.get("hardware"), str) else None
                )
                for mapping in pass_mapping_artifacts
            ):
                findings.append(
                    cross_finding(
                        "sim_cgra_cycle_requires_hardware",
                        (
                            f"sim kernel {kernel!r} has CGRA-sim cycles "
                            "but no pass mapping artifact references verified ADG hardware"
                        ),
                        row,
                    )
                )
    if workloads:
        for row in grouped.get("rtl_fpa", []):
            workload = row.get("workload")
            if valid_identity(workload) and workload not in workloads:
                findings.append(
                    cross_finding(
                        "rtl_fpa_workload_resolves",
                        f"RTL/FPA workload {workload!r} is absent from dataflow primitive coverage",
                        row,
                    )
                )

    if hardware:
        for row in pnr_rows:
            if canonical_hardware_ref(row.get("hardware")) is None:
                findings.append(
                    cross_finding(
                        "pnr_hardware_resolves",
                        f"PnR hardware {row['hardware']!r} is absent from verified ADG hardware summary",
                        row,
                    )
                )
        for row in grouped.get("rtl_fpa", []):
            candidate = row.get("hardware")
            if valid_identity(candidate) and candidate not in hardware:
                findings.append(
                    cross_finding(
                        "rtl_fpa_hardware_resolves",
                        f"RTL/FPA hardware {candidate!r} is absent from verified ADG hardware summary",
                        row,
                    )
                )

    if pnr_pairs:
        for row in grouped.get("dse_candidate", []):
            workload = row.get("workload")
            candidate = row.get("hardware")
            if not valid_identity(workload) or not valid_identity(candidate):
                continue
            assert workload is not None
            assert candidate is not None
            mapping_id = row.get("mapping_id", "")
            pnr_matches = matching_pass_pnr_rows(workload, candidate, mapping_id)
            mapping_artifact_matches = matching_pass_mapping_artifacts(
                workload, candidate, mapping_id
            )
            if (
                (workload, candidate) not in pnr_pairs
                and not pnr_matches
                and not mapping_artifact_matches
            ):
                findings.append(
                    cross_finding(
                        "dse_candidate_resolves_to_pnr",
                        (
                            f"DSE candidate ({workload!r}, {candidate!r}) is absent from "
                            "PnR mapping summary and mapping artifacts"
                        ),
                        row,
                    )
                )
            if row.get("selection_status") not in {"selected", "pareto"}:
                continue
            if not valid_identity(mapping_id):
                findings.append(
                    cross_finding(
                        "dse_selected_requires_mapping_id",
                        f"DSE selected candidate {row.get('candidate')!r} has no mapping_id",
                        row,
                    )
                )
                continue
            if len(pnr_matches) > 1:
                findings.append(
                    cross_finding(
                        "dse_selected_matches_pnr",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            f"matches {len(pnr_matches)} pass PnR rows"
                        ),
                        row,
                    )
                )
            if len(mapping_artifact_matches) != 1:
                findings.append(
                    cross_finding(
                        "dse_selected_matches_mapping_artifact",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            f"matches {len(mapping_artifact_matches)} pass PnR mapping artifacts"
                        ),
                        row,
                    )
                )
            sim_matches = matching_pass_sim_rows(workload)
            if len(sim_matches) != 1:
                findings.append(
                    cross_finding(
                        "dse_selected_matches_sim",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            f"matches {len(sim_matches)} pass simulator rows"
                        ),
                        row,
                    )
                )
                continue
            cgra_report_matches = matching_pass_cgra_reports(workload, candidate, mapping_id)
            if len(cgra_report_matches) != 1:
                findings.append(
                    cross_finding(
                        "dse_selected_matches_cgra_report",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            f"matches {len(cgra_report_matches)} pass CGRA reports"
                        ),
                        row,
                    )
                )
                continue
            fpa_matches = matching_pass_fpa_rows(workload, candidate)
            if len(fpa_matches) != 1:
                findings.append(
                    cross_finding(
                        "dse_selected_matches_fpa",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            f"matches {len(fpa_matches)} pass RTL/FPA rows"
                        ),
                        row,
                    )
                )
                continue
            sim_row = sim_matches[0]
            fpa_row = fpa_matches[0]
            cgra_report = cgra_report_matches[0]
            input_paths = referenced_artifact_paths(row, "input_artifacts")
            expected_inputs = (
                ("PnR summary", artifact_path(pnr_matches[0]) if len(pnr_matches) == 1 else None),
                (
                    "PnR mapping artifact",
                    artifact_path(mapping_artifact_matches[0])
                    if len(mapping_artifact_matches) == 1
                    else None,
                ),
                ("sim summary", artifact_path(sim_row)),
                ("CGRA report", artifact_path(cgra_report)),
                ("RTL/FPA summary", artifact_path(fpa_row)),
            )
            for label, expected_path in expected_inputs:
                if expected_path is not None and expected_path not in input_paths:
                    findings.append(
                        cross_finding(
                            "dse_selected_input_artifacts_match",
                            (
                                f"DSE selected candidate {row.get('candidate')!r} "
                                f"does not cite matched {label}"
                            ),
                            row,
                        )
                    )
            cgra_cycles = nonnegative_int_cell(row, "cgra_sim_cycles")
            sim_cycles = nonnegative_int_cell(sim_row, "cgra_sim_cycles")
            if cgra_cycles is None or sim_cycles is None or cgra_cycles != sim_cycles:
                findings.append(
                    cross_finding(
                        "dse_selected_cgra_cycle_matches_sim",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            "does not match simulator CGRA cycles"
                        ),
                        row,
                    )
                )
            report_cycles = cgra_report.get("hardware_aware_cycles")
            if not isinstance(report_cycles, int) or cgra_cycles != report_cycles:
                findings.append(
                    cross_finding(
                        "dse_selected_cgra_cycle_matches_cgra_report",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            "does not match CGRA report hardware-aware cycles"
                        ),
                        row,
                    )
                )
            for column in ("frequency_mhz", "area_um2", "dynamic_power_mw"):
                dse_value = float_cell(row, column)
                fpa_value = float_cell(fpa_row, column)
                if dse_value is None or fpa_value is None or not nearly_equal(dse_value, fpa_value):
                    findings.append(
                        cross_finding(
                            f"dse_selected_{column}_matches_fpa",
                            (
                                f"DSE selected candidate {row.get('candidate')!r} "
                                f"does not match RTL/FPA {column}"
                            ),
                            row,
                        )
                    )
            frequency = float_cell(fpa_row, "frequency_mhz")
            dynamic_power = float_cell(fpa_row, "dynamic_power_mw")
            leakage_power = float_cell(fpa_row, "leakage_power_mw")
            energy = float_cell(row, "energy_nj")
            if (
                cgra_cycles is None
                or frequency is None
                or frequency <= 0
                or dynamic_power is None
                or leakage_power is None
                or energy is None
            ):
                findings.append(
                    cross_finding(
                        "dse_selected_energy_inputs_present",
                        (
                            f"DSE selected candidate {row.get('candidate')!r} "
                            "lacks complete cycle/frequency/power inputs"
                        ),
                        row,
                    )
                )
            else:
                expected_energy = (dynamic_power + leakage_power) * cgra_cycles / frequency
                if not nearly_equal(energy, expected_energy):
                    findings.append(
                        cross_finding(
                            "dse_selected_energy_matches_fpa_formula",
                            (
                                f"DSE selected candidate {row.get('candidate')!r} "
                                f"energy {energy} does not match expected {expected_energy:.3f}"
                            ),
                            row,
                        )
                    )

    return findings


def cross_artifact_checks(paths: Iterable[Path]) -> list[dict[str, object]]:
    path_list = list(paths)
    grouped = rows_by_kind(path_list)
    json_grouped = json_objects_by_kind(path_list)
    checks: list[dict[str, object]] = []

    dfg_reports_by_workload = build_pass_dfg_reports_by_workload(json_grouped)
    cgra_reports_by_workload = build_pass_cgra_reports_by_workload(json_grouped)
    pass_mapping_artifacts_by_workload = build_pass_mapping_artifacts_by_workload(json_grouped)

    for row in grouped.get("sim_cycle", []):
        workload = row.get("kernel")
        if not valid_identity(workload):
            continue
        assert workload is not None
        dfg_cycles = nonnegative_int_cell(row, "dfg_sim_cycles")
        if dfg_cycles is None:
            continue
        dfg_reports = dfg_reports_by_workload.get(workload, [])
        dfg_report_cycles = [
            int(report["optimistic_cycles"])
            for report in dfg_reports
            if isinstance(report.get("optimistic_cycles"), int)
        ]
        if not dfg_report_cycles or sum(dfg_report_cycles) != dfg_cycles:
            continue
        graphs = sorted(
            str(report.get("graph"))
            for report in dfg_reports
            if isinstance(report.get("graph"), str)
        )
        dynamic_work_items = sum(
            int(report["dynamic_work_items"])
            for report in dfg_reports
            if isinstance(report.get("dynamic_work_items"), int)
        )
        checks.append(
            {
                "rule": "sim_cycle_dfg_report_evidence",
                "status": "pass",
                "workload": workload,
                "dfg_sim_cycles": dfg_cycles,
                "dfg_report_cycles": dfg_report_cycles,
                "graphs": graphs,
                "dynamic_work_items": dynamic_work_items,
            }
        )

        cgra_cycles = nonnegative_int_cell(row, "cgra_sim_cycles")
        if cgra_cycles is None:
            continue
        cgra_reports = cgra_reports_by_workload.get(workload, [])
        cgra_report_cycles = [
            int(report["hardware_aware_cycles"])
            for report in cgra_reports
            if isinstance(report.get("hardware_aware_cycles"), int)
        ]
        if not cgra_report_cycles or sum(cgra_report_cycles) != cgra_cycles:
            continue
        mapping_ids: list[str] = []
        component_mapping_ids: list[str] = []
        hardware_refs: list[str] = []
        route_segments = 0
        performance_delta_cycles = 0
        matched_all_mappings = True
        pass_mappings = pass_mapping_artifacts_by_workload.get(workload, [])
        for report in cgra_reports:
            mapping_id = report.get("mapping_id")
            hardware = report.get("hardware")
            if not isinstance(mapping_id, str) or not isinstance(hardware, str):
                matched_all_mappings = False
                break
            matching_mapping = None
            for mapping in pass_mappings:
                if mapping.get("mapping_id") != mapping_id or mapping.get("hardware") != hardware:
                    continue
                expected_route_segments = report.get("route_segments")
                if isinstance(expected_route_segments, int):
                    actual_route_segments = route_segment_count(mapping.get("routes"))
                    if (
                        actual_route_segments is not None
                        and actual_route_segments != expected_route_segments
                    ):
                        continue
                expected_config_records = report.get("config_records")
                if (
                    isinstance(expected_config_records, int)
                    and mapping.get("config_records") != expected_config_records
                ):
                    continue
                matching_mapping = mapping
                break
            if matching_mapping is None:
                matched_all_mappings = False
                break
            mapping_ids.append(mapping_id)
            for component_mapping_id in aggregate_component_identities(report, "component_mapping_ids"):
                component_mapping_ids.append(component_mapping_id)
            hardware_refs.append(hardware)
            if isinstance(report.get("route_segments"), int):
                route_segments += int(report["route_segments"])
            if isinstance(report.get("performance_delta_cycles"), int):
                performance_delta_cycles += int(report["performance_delta_cycles"])
        if not matched_all_mappings:
            continue
        checks.append(
            {
                "rule": "sim_cycle_report_mapping_evidence",
                "status": "pass",
                "workload": workload,
                "dfg_sim_cycles": dfg_cycles,
                "cgra_sim_cycles": cgra_cycles,
                "dfg_report_cycles": dfg_report_cycles,
                "cgra_report_cycles": cgra_report_cycles,
                "dynamic_work_items": dynamic_work_items,
                "mapping_ids": sorted(mapping_ids),
                "component_mapping_ids": sorted(set(component_mapping_ids)),
                "hardware": sorted(set(hardware_refs)),
                "route_segments": route_segments,
                "performance_delta_cycles": performance_delta_cycles,
            }
        )

    return checks


def audit(paths: Iterable[Path]) -> dict[str, object]:
    path_list = list(paths)
    reviews: list[dict[str, object]] = []
    diagnostics: list[str] = []
    for path in path_list:
        csv_schema = schema_for_path(path)
        if csv_schema is not None:
            review = audit_csv(path, csv_schema)
        else:
            json_kind = json_kind_for_path(path)
            if json_kind is None:
                review = {
                    "artifact": str(path),
                    "schema": "unknown",
                    "rows_checked": 0,
                    "parser_checks": [],
                    "finding": "fail",
                    "diagnostics": ["unknown artifact schema"],
                }
            else:
                review = audit_json(path, json_kind)
        reviews.append(review)
        if review["finding"] != "pass":
            diagnostics.extend(str(item) for item in review.get("diagnostics", []))
    cross_checks = cross_artifact_checks(path_list)
    cross_findings = cross_artifact_findings(path_list)
    diagnostics.extend(str(item["message"]) for item in cross_findings)
    return {
        "schema_version": 1,
        "run_id": "scaffold-audit",
        "artifact_reviews": reviews,
        "cross_artifact_checks": cross_checks,
        "cross_artifact_findings": cross_findings,
        "diagnostics": diagnostics,
        "verdict": "pass" if not diagnostics else "fail",
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    write_csv_parser = sub.add_parser("write-csv")
    write_csv_parser.add_argument("kind", choices=sorted(CSV_SCHEMAS))
    write_csv_parser.add_argument("--output", required=True)

    write_json_parser = sub.add_parser("write-json")
    write_json_parser.add_argument("kind", choices=["artifact_manifest"])
    write_json_parser.add_argument("--output", required=True)

    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument("artifacts", nargs="+")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "write-csv":
        write_csv(args.kind, output_path(args.output))
        return 0
    if args.command == "write-json":
        write_json(args.kind, output_path(args.output))
        return 0
    if args.command == "audit":
        result = audit(Path(path) for path in args.artifacts)
        output_path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0 if result["verdict"] == "pass" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

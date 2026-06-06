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
            "energy_nj",
            "selection_status",
        ),
        status_columns=("selection_status",),
        extra_columns=(
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
        numeric_columns=("cgra_sim_cycles", "frequency_mhz", "area_um2", "dynamic_power_mw", "energy_nj"),
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
            "blocked",
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
            "report_status",
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
            "referenced_workload_report_bundle_identities",
            "referenced_hardware_candidate_report_bundle_identities",
            "runtime_evidence_summaries",
            "selected_policy_id",
            "policy_configuration",
            "candidate_ordering_rule",
            "report_status",
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
    if objective in {"minimize_energy", "minimize_power"}:
        return "energy_score_then_candidate_id"
    return "runtime_score_then_candidate_id"


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
        if node_count is not None and node_count <= 0:
            diagnostics.append(f"row {row_index}: ADG hardware pass row has no nodes")
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


def artifact_reference_exists(anchor: Path, reference: str) -> bool:
    path = Path(reference)
    if path.is_absolute():
        return path.is_file()
    return path.is_file() or (anchor.parent / path).is_file()


def resolve_artifact_reference(anchor: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    if path.is_file():
        return path.resolve()
    return (anchor.parent / path).resolve()


def artifact_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_artifact_manifest_edges(data: dict[str, object], diagnostics: list[str]) -> int:
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

    for left, right in ARTIFACT_EDGE_PAIRS:
        if left in artifact_ids and right in artifact_ids:
            require_manifest_edge(edge_pairs, diagnostics, left, right)

    for mapping_id in ids_by_kind.get("pnr_mapping_artifact", []):
        for source_kind in ("dataflow_primitive_coverage", "adg_hardware", "pnr_mapping"):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, mapping_id)
        for cgra_id in ids_by_kind.get("cgra_sim_report", []):
            require_manifest_edge(edge_pairs, diagnostics, mapping_id, cgra_id)
        for dse_id in ids_by_kind.get("dse_candidate", []):
            require_manifest_edge(edge_pairs, diagnostics, mapping_id, dse_id)

    for dfg_id in ids_by_kind.get("dfg_sim_report", []):
        for source_id in ids_by_kind.get("dataflow_primitive_coverage", []):
            require_manifest_edge(edge_pairs, diagnostics, source_id, dfg_id)
        for sim_id in ids_by_kind.get("sim_cycle", []):
            require_manifest_edge(edge_pairs, diagnostics, dfg_id, sim_id)

    for cgra_id in ids_by_kind.get("cgra_sim_report", []):
        for sim_id in ids_by_kind.get("sim_cycle", []):
            if sim_id == "sim-cycle-summary":
                require_manifest_edge(edge_pairs, diagnostics, cgra_id, sim_id)
        for dse_id in ids_by_kind.get("dse_candidate", []):
            require_manifest_edge(edge_pairs, diagnostics, cgra_id, dse_id)

    for comparison_id in ids_by_kind.get("sim_comparison_report", []):
        for source_kind in ("dfg_sim_report", "cgra_sim_report", "pnr_mapping_artifact"):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, comparison_id)

    for runtime_id in ids_by_kind.get("runtime_package", []):
        for source_kind in ("pnr_mapping_artifact", "cgra_sim_report", "sim_comparison_report"):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, runtime_id)

    for report_id in ids_by_kind.get("workload_report_bundle", []):
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
            "rtl_fpa",
            "dse_candidate",
        ):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, report_id)
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            require_manifest_edge(edge_pairs, diagnostics, report_id, demonstrator_id)

    for hardware_report_id in ids_by_kind.get("hardware_report_bundle", []):
        for source_kind in ("adg_hardware", "rtl_fpa"):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, hardware_report_id)
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            require_manifest_edge(edge_pairs, diagnostics, hardware_report_id, demonstrator_id)

    for dse_report_id in ids_by_kind.get("dse_report_bundle", []):
        for source_kind in ("dse_candidate", "workload_report_bundle", "hardware_report_bundle"):
            for source_id in ids_by_kind.get(source_kind, []):
                require_manifest_edge(edge_pairs, diagnostics, source_id, dse_report_id)

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


def validate_runtime_handle_model(value: object, diagnostics: list[str]) -> None:
    if not isinstance(value, dict):
        diagnostics.append("runtime package runtime_handle_model must be an object")
        return
    if value.get("handle_kind") != "host_visible_launch_handle":
        diagnostics.append("runtime package runtime_handle_model handle_kind must be host_visible_launch_handle")
    if value.get("ir_token_kind") != "not_dataflow_thread_token":
        diagnostics.append("runtime package runtime_handle_model must not use dataflow thread tokens")
    if not isinstance(value.get("completion_source"), str) or not value.get("completion_source"):
        diagnostics.append("runtime package runtime_handle_model lacks completion_source")
    operations = value.get("operations")
    if not isinstance(operations, list) or any(not isinstance(operation, str) for operation in operations):
        diagnostics.append("runtime package runtime_handle_model operations must be a string list")
        return
    for operation in ("query_status", "wait_for_completion", "collect_diagnostics"):
        if operation not in operations:
            diagnostics.append(f"runtime package runtime_handle_model lacks {operation}")


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
    expected_configuration_id = (
        f"runtime-config::{data.get('fallback_policy')}::"
        f"{data.get('data_movement_policy')}::{data.get('synchronization_mode')}"
    )
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
    if "host_program_identity" in data and value.get("host_program_identity") != data.get("host_program_identity"):
        diagnostics.append(f"{label} host_interface host_program_identity does not match package")
    if "host_wrapper_identity" in data and value.get("host_wrapper_identity") != data.get("host_wrapper_identity"):
        diagnostics.append(f"{label} host_interface host_wrapper_identity does not match package")
    if value.get("compatibility_mode_requires_runtime") is not False:
        diagnostics.append(f"{label} compatibility mode must not require runtime")
    if value.get("acceleration_mode_requires_runtime_package") is not True:
        diagnostics.append(f"{label} acceleration mode must require runtime package")


def validate_report_only_runtime_claims(
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
    validate_diagnostic_records(value.get("diagnostic_records"), diagnostics, "runtime package runtime_report")
    if data.get("fallback_policy") == "report_only":
        validate_report_only_runtime_claims(
            value,
            output_buffers,
            diagnostics,
            "runtime package report_only runtime_report",
        )


def validate_runtime_evidence(value: object, diagnostics: list[str], require_complete: bool) -> None:
    if not isinstance(value, dict):
        diagnostics.append("workload report bundle runtime_evidence must be an object")
        return
    required_keys = (
        "runtime_package_identity",
        "runtime_report_identity",
        "launch_status",
        "target_status",
        "runtime_trace_identity",
        "profiling_record_identity",
        "data_movement_policy",
        "synchronization_mode",
        "output_buffer_identities",
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
    data_movement_policy = value.get("data_movement_policy")
    if data_movement_policy not in DATA_MOVEMENT_POLICIES:
        diagnostics.append("workload report bundle runtime_evidence has unknown data_movement_policy")
    outputs = value.get("output_buffer_identities")
    if not isinstance(outputs, list) or any(not isinstance(identity, str) for identity in outputs):
        diagnostics.append("workload report bundle runtime_evidence output_buffer_identities must be a string list")
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
    if data_movement_policy not in required_data_movement_policies:
        diagnostics.append(
            "workload report bundle runtime_evidence required_data_movement_policies omits data_movement_policy"
        )
    synchronization_mode = value.get("synchronization_mode")
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
    fallback = value.get("fallback_decision")
    if not isinstance(fallback, dict):
        diagnostics.append("workload report bundle runtime_evidence fallback_decision must be an object")
        fallback = {}
    if require_complete and not value.get("runtime_report_identity"):
        diagnostics.append("workload report bundle pass needs runtime_report_identity")
    if require_complete and not input_fingerprints:
        diagnostics.append("workload report bundle pass needs runtime input_artifact_fingerprints")
    if fallback.get("decision") == "report_only":
        validate_report_only_runtime_claims(
            value,
            outputs,
            diagnostics,
            "workload report bundle report_only runtime evidence",
        )


def validate_runtime_evidence_summaries(
    value: object,
    diagnostics: list[str],
    require_complete: bool,
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
        for key in (
            "workload_report_bundle_identity",
            "runtime_package_identity",
            "runtime_report_identity",
            "launch_status",
            "target_status",
            "data_movement_policy",
            "synchronization_mode",
        ):
            if not isinstance(summary.get(key), str) or not summary.get(key):
                diagnostics.append(f"DSE report bundle runtime evidence summary {index} lacks {key}")
        data_movement_policy = summary.get("data_movement_policy")
        if data_movement_policy not in DATA_MOVEMENT_POLICIES:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} has unknown data_movement_policy"
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
        if data_movement_policy not in required_data_movement_policies:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_data_movement_policies omits data_movement_policy"
            )
        synchronization_mode = summary.get("synchronization_mode")
        if synchronization_mode not in required_synchronization_policies:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} "
                "required_synchronization_policies omits synchronization_mode"
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
        if require_complete and not input_fingerprints:
            diagnostics.append(
                f"DSE report bundle runtime evidence summary {index} needs input_artifact_fingerprints"
            )
        fallback = summary.get("fallback_decision")
        validate_fallback_decision(
            fallback,
            diagnostics,
            f"DSE report bundle runtime evidence summary {index}",
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
        resolved = resolve_artifact_reference(path, identity)
        if resolved.is_file() and fingerprint != artifact_fingerprint(resolved):
            diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints stale for {identity!r}")
    for reference in input_refs:
        if reference not in input_fingerprints:
            diagnostics.append(f"DSE report bundle candidate {index} input_artifact_fingerprints lacks {reference!r}")


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
        manifest_entries_checked = validate_artifact_manifest_edges(data, diagnostics)
    if kind == "artifact_audit" and data.get("verdict") not in {"pass", "fail"}:
        diagnostics.append("artifact audit verdict must be pass or fail")
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
        data_movement_policy = data.get("data_movement_policy")
        if data_movement_policy not in DATA_MOVEMENT_POLICIES:
            diagnostics.append("runtime package has unknown data_movement_policy")
        diagnostics_list = data.get("diagnostics")
        if not isinstance(diagnostics_list, list):
            diagnostics.append("runtime package diagnostics must be a list")
        diagnostic_records = validate_diagnostic_records(
            data.get("diagnostic_records"),
            diagnostics,
            "runtime package",
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
        for index, descriptor in enumerate(memory_descriptors, start=1):
            if not isinstance(descriptor, dict):
                diagnostics.append(f"runtime package memory descriptor {index} must be an object")
                continue
            for key in (
                "logical_argument",
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
            if descriptor.get("transfer_policy") != data_movement_policy:
                diagnostics.append(
                    f"runtime package memory descriptor {index} transfer_policy does not match data_movement_policy"
                )
            platform_binding = descriptor.get("platform_binding_identity")
            if platform_binding is not None and (not isinstance(platform_binding, str) or not platform_binding):
                diagnostics.append(
                    f"runtime package memory descriptor {index} has invalid platform_binding_identity"
                )
        for index, descriptor in enumerate(argument_descriptors, start=1):
            if not isinstance(descriptor, dict):
                diagnostics.append(f"runtime package argument descriptor {index} must be an object")
                continue
            for key in ("name", "identity", "descriptor_kind"):
                if not isinstance(descriptor.get(key), str) or not descriptor.get(key):
                    diagnostics.append(f"runtime package argument descriptor {index} lacks {key}")
        if data_movement_policy not in required_data_movement_policies:
            diagnostics.append("runtime package required_data_movement_policies omits data_movement_policy")
        synchronization_mode = data.get("synchronization_mode")
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
        validate_host_interface(
            data.get("runtime_host_interface"),
            {},
            diagnostics,
            "workload report bundle runtime",
        )
        validate_runtime_evidence(
            data.get("runtime_evidence"),
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
            value = metric.get("value")
            if not isinstance(value, (int, float)) or value < 0:
                diagnostics.append(f"workload report bundle metric {index} has invalid value")
            metric_diagnostics = metric.get("diagnostics")
            if not isinstance(metric_diagnostics, list):
                diagnostics.append(f"workload report bundle metric {index} diagnostics must be a list")
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
    if kind == "hardware_report_bundle":
        if data.get("kind") != "hardware_report_bundle":
            diagnostics.append("hardware report bundle kind must be hardware_report_bundle")
        if data.get("report_status") not in BASE_STATUSES:
            diagnostics.append("hardware report bundle report_status must be a known status")
        for key in ("bundle_id", "hardware_candidate_identity", "fabric_adg_identity"):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"hardware report bundle lacks {key}")
        for key in (
            "eda_report_identities",
            "fpa_report_identities",
            "supported_workload_classes",
            "diagnostics",
        ):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"hardware report bundle {key} must be a list")
        if data.get("report_status") == "pass":
            if not data.get("fpa_report_identities"):
                diagnostics.append("hardware report bundle pass needs FPA report identity")
            if not data.get("supported_workload_classes"):
                diagnostics.append("hardware report bundle pass needs supported workload classes")
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
        for key in ("dse_run_id", "selected_policy_id", "candidate_ordering_rule"):
            if not isinstance(data.get(key), str) or not data.get(key):
                diagnostics.append(f"DSE report bundle lacks {key}")
        for key in (
            "objective_records",
            "candidate_list",
            "selected_candidates",
            "pareto_set",
            "rejected_candidate_summaries",
            "referenced_workload_report_bundle_identities",
            "referenced_hardware_candidate_report_bundle_identities",
            "runtime_evidence_summaries",
            "diagnostics",
        ):
            if not isinstance(data.get(key), list):
                diagnostics.append(f"DSE report bundle {key} must be a list")
        if not isinstance(data.get("policy_configuration"), dict):
            diagnostics.append("DSE report bundle policy_configuration must be an object")
        objectives = data.get("objective_records")
        if not isinstance(objectives, list) or not objectives:
            diagnostics.append("DSE report bundle needs non-empty objective_records")
            objectives = []
        for index, objective in enumerate(objectives, start=1):
            if not isinstance(objective, dict):
                diagnostics.append(f"DSE report bundle objective {index} must be an object")
                continue
            for key in (
                "objective_id",
                "objective_kind",
                "comparison_direction",
                "units",
            ):
                if not isinstance(objective.get(key), str) or not objective.get(key):
                    diagnostics.append(f"DSE report bundle objective {index} lacks {key}")
            for key in ("metric_inputs", "validity_conditions"):
                if not isinstance(objective.get(key), list) or not objective.get(key):
                    diagnostics.append(f"DSE report bundle objective {index} lacks {key}")
            priority = objective.get("priority")
            if not isinstance(priority, (int, float)) or priority <= 0:
                diagnostics.append(f"DSE report bundle objective {index} has invalid priority")
        candidates = data.get("candidate_list")
        candidate_ids: set[str] = set()
        if not isinstance(candidates, list) or not candidates:
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
            for key in ("candidate_kind", "status"):
                if not isinstance(candidate.get(key), str) or not candidate.get(key):
                    diagnostics.append(f"DSE report bundle candidate {index} lacks {key}")
            if candidate.get("status") not in SELECTION_STATUSES:
                diagnostics.append(f"DSE report bundle candidate {index} has unknown status")
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
            validate_dse_report_candidate_input_fingerprints(path, candidate, diagnostics, index)
        selected_candidates = data.get("selected_candidates")
        if isinstance(selected_candidates, list):
            missing_selected = [
                candidate_id
                for candidate_id in selected_candidates
                if candidate_id not in candidate_ids
            ]
            if missing_selected:
                diagnostics.append(f"DSE report bundle selected candidates are missing records {missing_selected}")
        if data.get("report_status") == "pass":
            for key in (
                "selected_candidates",
                "referenced_workload_report_bundle_identities",
                "referenced_hardware_candidate_report_bundle_identities",
            ):
                if not data.get(key):
                    diagnostics.append(f"DSE report bundle pass needs {key}")
        validate_runtime_evidence_summaries(
            data.get("runtime_evidence_summaries"),
            diagnostics,
            data.get("report_status") == "pass",
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
    return grouped


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
    return grouped


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

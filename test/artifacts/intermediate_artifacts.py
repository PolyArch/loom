#!/usr/bin/env python3
"""Intermediate artifact schema writers and deterministic content audit."""

from __future__ import annotations

import argparse
import csv
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
        extra_columns=("diagnostic",),
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
}


STANDARD_ARTIFACT_PATHS = (
    ("old_app_corpus_inventory", "temp/old-app-corpus-inventory.csv"),
    ("app_import_status", "temp/app-corpus-import-status.csv"),
    ("source_compat", "temp/source-compat-summary.csv"),
    ("compiler_pipeline", "temp/compiler-pipeline-summary.csv"),
    ("dataflow_primitive_coverage", "temp/dataflow-primitive-coverage.csv"),
    ("adg_hardware", "temp/adg-hardware-summary.csv"),
    ("pnr_mapping", "temp/pnr-mapping-summary.csv"),
    ("sim_cycle", "temp/sim-cycle-summary.csv"),
    ("rtl_fpa", "temp/rtl-fpa-summary.csv"),
    ("e2e_demonstrator", "temp/e2e-demonstrator-summary.csv"),
    ("dse_candidate", "temp/dse-candidate-summary.csv"),
    ("unsupported_scope", "temp/unsupported-scope-ledger.csv"),
)


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
    if embedded_kind == "dfg_sim_report":
        return "dfg_sim_report"
    if embedded_kind == "cgra_sim_report":
        return "cgra_sim_report"
    if embedded_kind == "pnr_mapping":
        return "pnr_mapping_artifact"
    return None


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
    return {
        "artifact": str(path),
        "schema": schema.kind,
        "rows_checked": len(rows),
        "parser_checks": ["csv_header", "status_values", "numeric_policy", "identity_policy"],
        "finding": "pass" if not diagnostics else "fail",
        "diagnostics": diagnostics,
    }


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
    return {
        "artifact": str(path),
        "schema": kind,
        "entries_checked": len(data) if isinstance(data, dict) else 0,
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
        grouped.setdefault(schema.kind, []).extend(read_csv_rows(path))
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
    pass_mapping_artifacts_by_workload: dict[str, list[dict[str, object]]] = {}
    for artifact in json_grouped.get("pnr_mapping_artifact", []):
        workload = artifact.get("workload")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and artifact.get("status") == "pass"
            and isinstance(artifact.get("mapping_id"), str)
            and isinstance(artifact.get("hardware"), str)
        ):
            pass_mapping_artifacts_by_workload.setdefault(workload, []).append(artifact)
    dfg_report_cycles_by_workload: dict[str, list[int]] = {}
    dfg_report_semantics_by_workload: dict[str, set[str]] = {}
    dfg_reports_by_workload_graph: dict[tuple[str, str], list[dict[str, object]]] = {}
    for report in json_grouped.get("dfg_sim_report", []):
        workload = report.get("workload")
        graph = report.get("graph")
        cycles = report.get("optimistic_cycles")
        semantics = report.get("operation_semantics_source")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and report.get("status") == "pass"
            and isinstance(cycles, int)
            and cycles >= 0
        ):
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
    cgra_reports_by_workload: dict[str, list[dict[str, object]]] = {}
    for report in json_grouped.get("cgra_sim_report", []):
        workload = report.get("workload")
        if (
            isinstance(workload, str)
            and valid_identity(workload)
            and report.get("status") == "pass"
        ):
            cgra_reports_by_workload.setdefault(workload, []).append(report)

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
                                    routes = mapping.get("routes")
                                    if isinstance(routes, list):
                                        mapped_segments = 0
                                        for route in routes:
                                            if isinstance(route, dict) and isinstance(route.get("segments"), list):
                                                mapped_segments += len(route["segments"])
                                        if mapped_segments != route_segments:
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
            if (workload, candidate) not in pnr_pairs:
                findings.append(
                    cross_finding(
                        "dse_candidate_resolves_to_pnr",
                        f"DSE candidate ({workload!r}, {candidate!r}) is absent from PnR mapping summary",
                        row,
                    )
                )

    return findings


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
    cross_findings = cross_artifact_findings(path_list)
    diagnostics.extend(str(item["message"]) for item in cross_findings)
    return {
        "schema_version": 1,
        "run_id": "scaffold-audit",
        "artifact_reviews": reviews,
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

#!/usr/bin/env python3
"""Intermediate artifact schema writers and deterministic content audit."""

from __future__ import annotations

import argparse
import csv
import json
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
            "status",
        ),
        status_columns=("status",),
        extra_columns=("diagnostic",),
        numeric_columns=("placed_records", "routed_edges", "unrouted_edges"),
        identity_columns=("workload", "hardware", "mapping_id"),
        scaffold_row=(
            "scaffold",
            "scaffold",
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
        if any(value == "pass" for _, value in row_statuses(schema, row)):
            diagnostic = row.get("diagnostic", "")
            if diagnostic == "" and schema.kind not in {"sim_cycle", "pnr_mapping"}:
                diagnostics.append(f"row {index}: pass row has no diagnostic or evidence note")
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
    grouped = rows_by_kind(paths)
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
    pnr_rows = [
        row
        for row in grouped.get("pnr_mapping", [])
        if valid_identity(row.get("workload")) and valid_identity(row.get("hardware"))
    ]
    pnr_pairs = {(row["workload"], row["hardware"]) for row in pnr_rows}

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
            if valid_identity(kernel) and kernel not in workloads:
                findings.append(
                    cross_finding(
                        "sim_workload_resolves",
                        f"sim kernel {kernel!r} is absent from dataflow primitive coverage",
                        row,
                    )
                )
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
            if row["hardware"] not in hardware:
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

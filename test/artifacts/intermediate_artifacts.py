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
        numeric_columns=("placed_records", "routed_edges", "unrouted_edges"),
        identity_columns=("workload", "hardware", "mapping_id"),
        scaffold_row=("scaffold", "scaffold", "", "0", "0", "0", "blocked"),
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


def schema_for_path(path: Path) -> CsvSchema | None:
    name = path.name
    for schema in CSV_SCHEMAS.values():
        if name == schema.filename or name.endswith("-" + schema.filename):
            return schema
    return None


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
        allowed = SELECTION_STATUSES if column == "selection_status" else BASE_STATUSES
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


def audit(paths: Iterable[Path]) -> dict[str, object]:
    reviews: list[dict[str, object]] = []
    diagnostics: list[str] = []
    for path in paths:
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
    return {
        "schema_version": 1,
        "run_id": "scaffold-audit",
        "artifact_reviews": reviews,
        "cross_artifact_findings": [],
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

#!/usr/bin/env python3
"""Audit row-complete CGRA status CSV and JSON evidence."""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))
sys.path.insert(0, str(ROOT / "test" / "e2e"))

import cgra_status_summary  # noqa: E402
import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--json-input")
    parser.add_argument(
        "--legacy-loombench-root",
        default=str(ROOT / "temp" / "old_implementation_loom" / "loom" / "tests" / "app"),
    )
    return parser.parse_args(argv)


def json_path_for(csv_input: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return csv_input.with_suffix(".json")


def read_csv_rows(path: Path, diagnostics: list[str]) -> list[dict[str, str]]:
    schema = intermediate_artifacts.CSV_SCHEMAS["cgra_status"]
    if not path.is_file():
        diagnostics.append(f"missing CGRA status CSV: {path}")
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        expected_header = list(schema.first_columns)
        if reader.fieldnames != expected_header:
            diagnostics.append(f"unexpected CGRA status header: {reader.fieldnames}")
    return rows


def expected_rows(legacy_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(cgra_status_summary.app_rows())
    rows.extend(
        cgra_status_summary.cmsis_rows(
            "cmsis-dsp",
            "cmsis-dsp",
            "cmsis_dsp_targets.txt",
            "externals/cmsis-dsp/Source",
        )
    )
    rows.extend(
        cgra_status_summary.cmsis_rows(
            "cmsis-nn",
            "cmsis-nn",
            "cmsis_nn_targets.txt",
            "externals/cmsis-nn/Source",
        )
    )
    rows.extend(cgra_status_summary.loombench_rows(legacy_root))
    return rows


def identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("suite", ""), row.get("case", ""), row.get("source_row", ""))


def fingerprint_is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)


def resolve_artifact_reference(csv_input: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    candidates = (
        path,
        csv_input.parent / path,
        ROOT / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (ROOT / path).resolve()


def validate_artifact_fingerprint(
    *,
    csv_input: Path,
    row_index: int,
    row: dict[str, str],
    artifact_column: str,
    fingerprint_column: str,
    diagnostics: list[str],
) -> None:
    raw_path = row.get(artifact_column, "")
    raw_fingerprint = row.get(fingerprint_column, "")
    if not raw_path:
        diagnostics.append(f"row {row_index}: pass row lacks {artifact_column}")
        return
    if not fingerprint_is_sha256(raw_fingerprint):
        diagnostics.append(f"row {row_index}: pass row has invalid {fingerprint_column}")
    artifact_path = resolve_artifact_reference(csv_input, raw_path)
    if not artifact_path.is_file():
        diagnostics.append(f"row {row_index}: pass row artifact path does not exist in {artifact_column}: {raw_path}")
        return
    if fingerprint_is_sha256(raw_fingerprint):
        actual = intermediate_artifacts.artifact_fingerprint(artifact_path)
        if raw_fingerprint != actual:
            diagnostics.append(
                f"row {row_index}: pass row {fingerprint_column} does not match {artifact_column}"
            )


def validate_rows(csv_input: Path, rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    allowed = intermediate_artifacts.BASE_STATUSES
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        row_id = identity(row)
        if row_id in seen:
            diagnostics.append(f"row {index}: duplicate row identity {row_id}")
        seen.add(row_id)
        for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status", "status"):
            if row.get(column, "") not in allowed:
                diagnostics.append(f"row {index}: {column} has invalid status {row.get(column)!r}")
        try:
            slice_count = int(row.get("required_slice_count", ""))
        except ValueError:
            diagnostics.append(f"row {index}: required_slice_count is not an integer")
            continue
        if slice_count < 0:
            diagnostics.append(f"row {index}: required_slice_count is negative")
        if row.get("status") != "pass":
            for column in ("diagnostic_class", "owner", "blocking_prerequisite", "diagnostic"):
                if not row.get(column, ""):
                    diagnostics.append(f"row {index}: non-pass row lacks {column}")
        if row.get("status") == "pass":
            for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
                if row.get(column, "") != "pass":
                    diagnostics.append(f"row {index}: pass row requires {column}=pass")
            if slice_count <= 0:
                diagnostics.append(f"row {index}: pass row requires positive required_slice_count")
            if row.get("final_outputs_present") != "true" and row.get("final_memory_state_present") != "true":
                diagnostics.append(f"row {index}: pass row lacks final output or final memory-state evidence")
            for artifact_column, fingerprint_column in (
                ("dfg_report", "dfg_report_fingerprint"),
                ("mapping_artifact", "mapping_artifact_fingerprint"),
                ("cgra_report", "cgra_report_fingerprint"),
                ("comparison_report", "comparison_report_fingerprint"),
            ):
                validate_artifact_fingerprint(
                    csv_input=csv_input,
                    row_index=index,
                    row=row,
                    artifact_column=artifact_column,
                    fingerprint_column=fingerprint_column,
                    diagnostics=diagnostics,
                )


def validate_coverage(rows: list[dict[str, str]], expected: list[dict[str, str]], diagnostics: list[str]) -> None:
    actual_ids = {identity(row) for row in rows}
    expected_ids = {identity(row) for row in expected}
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        diagnostics.append(f"row coverage mismatch: missing={missing[:10]} extra={extra[:10]}")


def validate_json(path: Path, csv_input: Path, rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    if not path.is_file():
        diagnostics.append(f"missing CGRA status JSON: {path}")
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        diagnostics.append(f"CGRA status JSON is invalid: {exc}")
        return
    if data.get("schema_version") != 1:
        diagnostics.append("CGRA status JSON schema_version must be 1")
    if data.get("kind") != "cgra_status_summary":
        diagnostics.append("CGRA status JSON kind must be cgra_status_summary")
    if data.get("csv_projection") != str(csv_input):
        diagnostics.append("CGRA status JSON csv_projection does not match audited CSV")
    json_rows = data.get("rows")
    if not isinstance(json_rows, list):
        diagnostics.append("CGRA status JSON rows must be a list")
        return
    typed_json_rows: list[dict[str, str]] = []
    for index, row in enumerate(json_rows):
        if not isinstance(row, dict):
            diagnostics.append(f"CGRA status JSON row {index} is not an object")
            continue
        typed_json_rows.append({str(key): str(value) for key, value in row.items()})
    csv_by_id = {identity(row): row for row in rows}
    json_by_id = {identity(row): row for row in typed_json_rows}
    if set(json_by_id) != set(csv_by_id):
        diagnostics.append("CGRA status JSON rows do not match CSV rows")
    for row_id, csv_row in csv_by_id.items():
        json_row = json_by_id.get(row_id)
        if json_row is not None and json_row != csv_row:
            diagnostics.append(f"CGRA status JSON row content does not match CSV row {row_id}")
            break
    if data.get("counts") != cgra_status_summary.suite_counts(rows):
        diagnostics.append("CGRA status JSON counts do not match CSV rows")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    diagnostics: list[str] = []
    csv_input = Path(args.input)
    rows = read_csv_rows(csv_input, diagnostics)
    expected = expected_rows(Path(args.legacy_loombench_root))
    validate_rows(csv_input, rows, diagnostics)
    validate_coverage(rows, expected, diagnostics)
    validate_json(json_path_for(csv_input, args.json_input), csv_input, rows, diagnostics)

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

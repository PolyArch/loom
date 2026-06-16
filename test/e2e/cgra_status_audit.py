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
    parser.add_argument("--loombench-manifest")
    parser.add_argument(
        "--no-legacy-loombench",
        action="store_true",
        help="Audit a status rollup that intentionally omits legacy LoomBench rows.",
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


def expected_rows(
    legacy_root: Path,
    loombench_manifest_path: Path | None,
    include_legacy_loombench: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    app_rows = cgra_status_summary.app_rows()
    rows.extend(app_rows)
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
    if include_legacy_loombench:
        rows.extend(
            cgra_status_summary.loombench_rows(
                legacy_root,
                loombench_manifest_path,
                {row["case"]: row for row in app_rows},
            )
        )
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


def validate_optional_artifact_fingerprint(
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
    if not raw_path and not raw_fingerprint:
        return
    if not raw_path:
        diagnostics.append(f"row {row_index}: referenced artifact row lacks {artifact_column}")
        return
    if not fingerprint_is_sha256(raw_fingerprint):
        diagnostics.append(f"row {row_index}: referenced artifact row has invalid {fingerprint_column}")
    artifact_path = resolve_artifact_reference(csv_input, raw_path)
    if not artifact_path.is_file():
        diagnostics.append(
            f"row {row_index}: referenced artifact path does not exist in {artifact_column}: {raw_path}"
        )
        return
    if fingerprint_is_sha256(raw_fingerprint):
        actual = intermediate_artifacts.artifact_fingerprint(artifact_path)
        if raw_fingerprint != actual:
            diagnostics.append(
                f"row {row_index}: referenced artifact row {fingerprint_column} does not match {artifact_column}"
            )


def read_referenced_json(csv_input: Path, row: dict[str, str], column: str) -> dict[str, object]:
    path = resolve_artifact_reference(csv_input, row.get(column, ""))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def valid_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def valid_memory_state(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) and valid_string_list(item) for key, item in value.items())


def validate_workload_identity(
    row_index: int,
    row: dict[str, str],
    label: str,
    data: dict[str, object],
    diagnostics: list[str],
) -> None:
    workload = data.get("workload")
    case = row.get("case", "")
    if not isinstance(workload, str) or not workload:
        diagnostics.append(f"row {row_index}: referenced {label} JSON lacks workload identity")
    elif workload != case:
        diagnostics.append(
            f"row {row_index}: referenced {label} JSON workload identity {workload!r} does not match row case {case!r}"
        )


def validate_pass_row_referenced_jsons(
    csv_input: Path,
    row_index: int,
    row: dict[str, str],
    diagnostics: list[str],
) -> None:
    dfg = read_referenced_json(csv_input, row, "dfg_report")
    mapping = read_referenced_json(csv_input, row, "mapping_artifact")
    cgra = read_referenced_json(csv_input, row, "cgra_report")
    comparison = read_referenced_json(csv_input, row, "comparison_report")
    for label, data, expected_kind in (
        ("dfg_report", dfg, "dfg_sim_report"),
        ("mapping_artifact", mapping, "pnr_mapping"),
        ("cgra_report", cgra, "cgra_sim_report"),
        ("comparison_report", comparison, "sim_comparison_report"),
    ):
        if data.get("kind") != expected_kind:
            diagnostics.append(f"row {row_index}: referenced {label} JSON kind is not {expected_kind}")
        if data.get("status") != "pass":
            diagnostics.append(f"row {row_index}: referenced {label} JSON status is not pass")
        validate_workload_identity(row_index, row, label, data, diagnostics)
        diagnostics.extend(
            intermediate_artifacts.cgra_status_graph_identity_diagnostics(
                row, label, data, row_index=row_index
            )
        )
    if comparison.get("functional_comparison_status") != "pass":
        diagnostics.append(f"row {row_index}: referenced comparison_report functional status is not pass")
    if comparison.get("memory_comparison_status") != "pass":
        diagnostics.append(f"row {row_index}: referenced comparison_report memory status is not pass")
    if comparison.get("performance_comparison_status") != "pass":
        diagnostics.append(f"row {row_index}: referenced comparison_report performance status is not pass")
    if cgra.get("functional_state_source") not in intermediate_artifacts.CGRA_FUNCTIONAL_STATE_SOURCES:
        diagnostics.append(f"row {row_index}: referenced cgra_report lacks final-state provenance")
    outputs_match = (
        valid_string_list(dfg.get("final_outputs"))
        and valid_string_list(cgra.get("final_outputs"))
        and dfg.get("final_outputs") == cgra.get("final_outputs")
    )
    memory_match = (
        valid_memory_state(dfg.get("final_memory_state"))
        and valid_memory_state(cgra.get("final_memory_state"))
        and dfg.get("final_memory_state") == cgra.get("final_memory_state")
    )
    if row.get("final_outputs_present") == "true" and not outputs_match:
        diagnostics.append(f"row {row_index}: final_outputs_present contradicts referenced reports")
    if row.get("final_memory_state_present") == "true" and not memory_match:
        diagnostics.append(f"row {row_index}: final_memory_state_present contradicts referenced reports")
    if not outputs_match and not memory_match:
        diagnostics.append(f"row {row_index}: referenced reports lack matching final state")


def validate_non_pass_row_referenced_jsons(
    csv_input: Path,
    row_index: int,
    row: dict[str, str],
    diagnostics: list[str],
) -> None:
    for label, expected_kind, status_column in (
        ("dfg_report", "dfg_sim_report", "dfg_status"),
        ("mapping_artifact", "pnr_mapping", "mapping_status"),
        ("cgra_report", "cgra_sim_report", "cgra_status"),
        ("comparison_report", "sim_comparison_report", "comparison_status"),
    ):
        if not row.get(label, ""):
            continue
        data = read_referenced_json(csv_input, row, label)
        if not data:
            diagnostics.append(f"row {row_index}: referenced {label} JSON is not parseable")
            continue
        if data.get("kind") != expected_kind:
            diagnostics.append(f"row {row_index}: referenced {label} JSON kind is not {expected_kind}")
        row_status = row.get(status_column, "")
        if row_status == "not_run":
            diagnostics.append(f"row {row_index}: referenced {label} requires {status_column} to match JSON status")
        elif data.get("status") != row_status:
            diagnostics.append(
                f"row {row_index}: referenced {label} JSON status does not match {status_column}"
            )
        diagnostics.extend(
            intermediate_artifacts.cgra_status_graph_identity_diagnostics(
                row, label, data, row_index=row_index
            )
        )
        if label == "dfg_report" and row.get("dfg_mlir", ""):
            input_fingerprints = data.get("input_artifact_fingerprints")
            if not isinstance(input_fingerprints, dict):
                diagnostics.append(f"row {row_index}: referenced dfg_report lacks input_artifact_fingerprints")
            else:
                dfg_mlir = resolve_artifact_reference(csv_input, row.get("dfg_mlir", ""))
                identity = intermediate_artifacts.artifact_id_for_path(dfg_mlir)
                expected = row.get("dfg_mlir_fingerprint", "")
                actual = input_fingerprints.get(identity)
                if actual != expected:
                    diagnostics.append(
                        f"row {row_index}: referenced dfg_report input_artifact_fingerprints stale for dfg_mlir"
                    )
        validate_workload_identity(row_index, row, label, data, diagnostics)


def validate_cmsis_dfg_mlir_evidence(
    csv_input: Path,
    row_index: int,
    row: dict[str, str],
    diagnostics: list[str],
) -> None:
    diagnostic_class = row.get("diagnostic_class", "")
    if diagnostic_class not in {
        "cmsis_dfg_mlir_missing",
        "cmsis_dfg_mlir_ready_for_dfg_sim",
        "cmsis_no_dataflow_graph",
        "cmsis_dfg_mlir_identity_mismatch",
    }:
        return
    if row.get("owner", "") != "compiler_pipeline":
        diagnostics.append(f"row {row_index}: CMSIS DFG MLIR evidence row owner must be compiler_pipeline")
    if row.get("dfg_report", "") or row.get("dfg_report_fingerprint", ""):
        diagnostics.append(f"row {row_index}: CMSIS DFG MLIR evidence must not be stored as DFG-sim report evidence")
    if diagnostic_class != "cmsis_dfg_mlir_missing":
        validate_artifact_fingerprint(
            csv_input=csv_input,
            row_index=row_index,
            row=row,
            artifact_column="dfg_mlir",
            fingerprint_column="dfg_mlir_fingerprint",
            diagnostics=diagnostics,
        )
    expected_name = f"{Path(row.get('source_row', '')).stem}.dfg.mlir"
    if row.get("dfg_mlir", "") and Path(row.get("dfg_mlir", "")).name != expected_name:
        diagnostics.append(f"row {row_index}: CMSIS dfg_mlir basename must be {expected_name}")
    resolved_dfg_mlir = resolve_artifact_reference(csv_input, row.get("dfg_mlir", ""))
    actual_graph_ids: list[str] = []
    if resolved_dfg_mlir.is_file():
        actual_graph_ids = cgra_status_summary.dfg_graph_ids_from_text(
            resolved_dfg_mlir.read_text(errors="replace")
        )
    row_graph_ids = [item for item in row.get("graph_ids", "").split(",") if item]
    for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row.get(column, "") != "not_run":
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR evidence row requires {column}=not_run")
    for column in (
        "mapping_artifact",
        "mapping_artifact_fingerprint",
        "cgra_report",
        "cgra_report_fingerprint",
        "comparison_report",
        "comparison_report_fingerprint",
    ):
        if row.get(column, ""):
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR evidence row must not carry {column}")
    if diagnostic_class == "cmsis_dfg_mlir_ready_for_dfg_sim":
        if row.get("status", "") != "blocked":
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR ready row requires status=blocked")
        if row.get("blocking_prerequisite", "") != "dfg_sim_report":
            diagnostics.append(
                f"row {row_index}: CMSIS DFG MLIR ready row requires blocking_prerequisite=dfg_sim_report"
            )
        if not row.get("graph_ids", ""):
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR ready row requires graph_ids")
        try:
            slice_count = int(row.get("required_slice_count", ""))
        except ValueError:
            slice_count = -1
        if slice_count <= 0 or slice_count != len(row_graph_ids):
            diagnostics.append(
                f"row {row_index}: CMSIS DFG MLIR ready row required_slice_count must match graph_ids count"
            )
        if not actual_graph_ids:
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR ready row requires dfg_mlir content graph_ids")
        elif row_graph_ids != actual_graph_ids:
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR ready row graph_ids must match dfg_mlir content")
    elif diagnostic_class == "cmsis_no_dataflow_graph":
        if row.get("status", "") != "unsupported":
            diagnostics.append(f"row {row_index}: CMSIS no-graph row requires status=unsupported")
        if row.get("blocking_prerequisite", "") != "dataflow_graph":
            diagnostics.append(f"row {row_index}: CMSIS no-graph row requires blocking_prerequisite=dataflow_graph")
        if row.get("required_slice_count", "") != "0":
            diagnostics.append(f"row {row_index}: CMSIS no-graph row requires required_slice_count=0")
        if row.get("graph_ids", ""):
            diagnostics.append(f"row {row_index}: CMSIS no-graph row requires empty graph_ids")
        if actual_graph_ids:
            diagnostics.append(f"row {row_index}: CMSIS no-graph row dfg_mlir must not contain dataflow graph ids")
    elif diagnostic_class == "cmsis_dfg_mlir_identity_mismatch":
        if row.get("status", "") != "fail":
            diagnostics.append(f"row {row_index}: CMSIS DFG MLIR identity mismatch row requires status=fail")
        if row.get("blocking_prerequisite", "") != "dataflow_graph_identity":
            diagnostics.append(
                f"row {row_index}: CMSIS DFG MLIR identity mismatch row requires "
                "blocking_prerequisite=dataflow_graph_identity"
            )
    elif diagnostic_class == "cmsis_dfg_mlir_missing":
        if row.get("status", "") != "blocked":
            diagnostics.append(f"row {row_index}: CMSIS missing DFG MLIR row requires status=blocked")
        if row.get("blocking_prerequisite", "") != "dfg_mlir":
            diagnostics.append(f"row {row_index}: CMSIS missing DFG MLIR row requires blocking_prerequisite=dfg_mlir")
        if row.get("graph_ids", ""):
            diagnostics.append(f"row {row_index}: CMSIS missing DFG MLIR row requires empty graph_ids")
        if row.get("dfg_mlir", "") or row.get("dfg_mlir_fingerprint", ""):
            diagnostics.append(f"row {row_index}: CMSIS missing DFG MLIR row must not carry dfg_mlir evidence")


def validate_cmsis_no_missing_status(row_index: int, row: dict[str, str], diagnostics: list[str]) -> None:
    if row.get("suite", "") not in {"cmsis-dsp", "cmsis-nn"}:
        return
    if row.get("diagnostic_class", "") == "missing_status":
        diagnostics.append(f"row {row_index}: CMSIS row must not use missing_status")


def validate_cmsis_dfg_mlir_reference(
    csv_input: Path,
    row_index: int,
    row: dict[str, str],
    diagnostics: list[str],
) -> None:
    if row.get("suite", "") not in {"cmsis-dsp", "cmsis-nn"}:
        return
    if not row.get("dfg_mlir", ""):
        return
    validate_artifact_fingerprint(
        csv_input=csv_input,
        row_index=row_index,
        row=row,
        artifact_column="dfg_mlir",
        fingerprint_column="dfg_mlir_fingerprint",
        diagnostics=diagnostics,
    )
    expected_name = f"{Path(row.get('source_row', '')).stem}.dfg.mlir"
    if Path(row.get("dfg_mlir", "")).name != expected_name:
        diagnostics.append(f"row {row_index}: CMSIS dfg_mlir basename must be {expected_name}")


def validate_cmsis_dfg_mlir_requirement(row_index: int, row: dict[str, str], diagnostics: list[str]) -> None:
    if row.get("suite", "") not in {"cmsis-dsp", "cmsis-nn"}:
        return
    has_dfg_mlir = bool(row.get("dfg_mlir", ""))
    if row.get("status", "") == "pass" and not has_dfg_mlir:
        diagnostics.append(f"row {row_index}: CMSIS pass row requires DFG MLIR evidence")
    if row.get("status", "") != "pass" and not has_dfg_mlir:
        if row.get("diagnostic_class", "") != "cmsis_dfg_mlir_missing":
            diagnostics.append(
                f"row {row_index}: CMSIS row without DFG MLIR evidence must use cmsis_dfg_mlir_missing"
            )


def validate_loombench_no_missing_status(row_index: int, row: dict[str, str], diagnostics: list[str]) -> None:
    if row.get("suite", "") != "loombench":
        return
    if row.get("diagnostic_class", "") == "missing_status":
        diagnostics.append(f"row {row_index}: LoomBench row must not use missing_status")


def validate_loombench_no_manifest_row(row_index: int, row: dict[str, str], diagnostics: list[str]) -> None:
    if row.get("suite", "") != "loombench":
        return
    if row.get("diagnostic_class", "") != "loombench_manifest_missing":
        return
    if row.get("status", "") != "blocked":
        diagnostics.append(f"row {row_index}: LoomBench row without manifest requires status=blocked")
    if row.get("owner", "") != "loombench_manifest":
        diagnostics.append(f"row {row_index}: LoomBench row without manifest requires owner=loombench_manifest")
    if row.get("blocking_prerequisite", "") != "loombench_manifest":
        diagnostics.append(
            f"row {row_index}: LoomBench row without manifest requires blocking_prerequisite=loombench_manifest"
        )
    if row_has_sim_artifacts(row):
        diagnostics.append(f"row {row_index}: LoomBench row without manifest must not carry simulator artifacts")


def app_manifest_no_dfg_cases(diagnostics: list[str]) -> set[str]:
    try:
        manifest = cgra_status_summary.load_app_manifest()
    except (SystemExit, OSError, json.JSONDecodeError) as exc:
        diagnostics.append(f"failed to load app manifest for no-DFG audit: {exc}")
        return set()
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        diagnostics.append("app manifest cases must be a list for no-DFG audit")
        return set()
    result: set[str] = set()
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case = entry.get("case")
        tiers = entry.get("tiers", [])
        if isinstance(case, str) and case and (not isinstance(tiers, list) or "dfg" not in tiers):
            result.add(case)
    return result


def validate_app_no_dfg_row(
    row_index: int,
    row: dict[str, str],
    diagnostics: list[str],
) -> None:
    if row.get("status", "") != "blocked":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires status=blocked")
    if row.get("diagnostic_class", "") != "app_dataflow_tier_missing":
        diagnostics.append(
            f"row {row_index}: app row without dfg tier requires diagnostic_class=app_dataflow_tier_missing"
        )
    if row.get("owner", "") != "compiler_pipeline":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires owner=compiler_pipeline")
    if row.get("blocking_prerequisite", "") != "dataflow":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires blocking_prerequisite=dataflow")
    if row.get("required_slice_count", "") != "0":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires required_slice_count=0")
    if row.get("graph_ids", ""):
        diagnostics.append(f"row {row_index}: app row without dfg tier requires empty graph_ids")
    for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row.get(column, "") != "not_run":
            diagnostics.append(f"row {row_index}: app row without dfg tier requires {column}=not_run")
    for column in (
        "dfg_mlir",
        "dfg_mlir_fingerprint",
        "dfg_report",
        "dfg_report_fingerprint",
        "mapping_artifact",
        "mapping_artifact_fingerprint",
        "cgra_report",
        "cgra_report_fingerprint",
        "comparison_report",
        "comparison_report_fingerprint",
    ):
        if row.get(column, ""):
            diagnostics.append(f"row {row_index}: app row without dfg tier must not carry {column}")
    if row.get("final_outputs_present", "") != "false":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires final_outputs_present=false")
    if row.get("final_memory_state_present", "") != "false":
        diagnostics.append(f"row {row_index}: app row without dfg tier requires final_memory_state_present=false")
    if "app manifest has no dfg tier" not in row.get("diagnostic", ""):
        diagnostics.append(f"row {row_index}: app row without dfg tier diagnostic must cite the app manifest")


def validate_rows(csv_input: Path, rows: list[dict[str, str]], diagnostics: list[str]) -> None:
    allowed = intermediate_artifacts.BASE_STATUSES
    seen: set[tuple[str, str, str]] = set()
    no_dfg_app_cases = app_manifest_no_dfg_cases(diagnostics)
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
            validate_pass_row_referenced_jsons(csv_input, index, row, diagnostics)
        else:
            for artifact_column, fingerprint_column in (
                ("dfg_report", "dfg_report_fingerprint"),
                ("mapping_artifact", "mapping_artifact_fingerprint"),
                ("cgra_report", "cgra_report_fingerprint"),
                ("comparison_report", "comparison_report_fingerprint"),
            ):
                validate_optional_artifact_fingerprint(
                    csv_input=csv_input,
                    row_index=index,
                    row=row,
                    artifact_column=artifact_column,
                    fingerprint_column=fingerprint_column,
                    diagnostics=diagnostics,
                )
            validate_non_pass_row_referenced_jsons(csv_input, index, row, diagnostics)
            validate_cmsis_dfg_mlir_evidence(csv_input, index, row, diagnostics)
        validate_cmsis_no_missing_status(index, row, diagnostics)
        validate_cmsis_dfg_mlir_reference(csv_input, index, row, diagnostics)
        validate_cmsis_dfg_mlir_requirement(index, row, diagnostics)
        validate_loombench_no_missing_status(index, row, diagnostics)
        validate_loombench_no_manifest_row(index, row, diagnostics)
        if row.get("suite", "") == "app" and row.get("case", "") in no_dfg_app_cases:
            validate_app_no_dfg_row(index, row, diagnostics)


def validate_coverage(rows: list[dict[str, str]], expected: list[dict[str, str]], diagnostics: list[str]) -> None:
    actual_ids = {identity(row) for row in rows}
    expected_ids = {identity(row) for row in expected}
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        diagnostics.append(f"row coverage mismatch: missing={missing[:10]} extra={extra[:10]}")


def load_loombench_manifest_map(path: Path | None, diagnostics: list[str]) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    try:
        cases = cgra_status_summary.load_loombench_manifest(path)
    except SystemExit as exc:
        diagnostics.append(str(exc))
        return {}
    return {str(case_data["case"]): case_data for case_data in cases}


def row_has_sim_artifacts(row: dict[str, str]) -> bool:
    return any(
        row.get(column, "")
        for column in (
            "dfg_report",
            "dfg_report_fingerprint",
            "mapping_artifact",
            "mapping_artifact_fingerprint",
            "cgra_report",
            "cgra_report_fingerprint",
            "comparison_report",
            "comparison_report_fingerprint",
        )
    )


def validate_loombench_manifest_semantics(
    rows: list[dict[str, str]],
    manifest_path: Path | None,
    diagnostics: list[str],
) -> None:
    if manifest_path is None:
        for row in rows:
            if row.get("suite", "") != "loombench":
                continue
            case = row.get("case", "")
            if row.get("status", "") != "blocked":
                diagnostics.append(f"LoomBench row without manifest {case} must stay blocked")
            if row.get("diagnostic_class", "") == "missing_status":
                diagnostics.append(f"LoomBench row without manifest must not use missing_status: {case}")
            elif row.get("diagnostic_class", "") != "loombench_manifest_missing":
                diagnostics.append(f"LoomBench row without manifest {case} has wrong diagnostic_class")
            if row.get("owner", "") != "loombench_manifest":
                diagnostics.append(f"LoomBench row without manifest {case} requires owner=loombench_manifest")
            if row.get("blocking_prerequisite", "") != "loombench_manifest":
                diagnostics.append(
                    f"LoomBench row without manifest {case} requires blocking_prerequisite=loombench_manifest"
                )
            if row_has_sim_artifacts(row):
                diagnostics.append(f"LoomBench row without manifest {case} must not carry simulator artifacts")
        return
    manifest_by_case = load_loombench_manifest_map(manifest_path, diagnostics)
    if not manifest_by_case:
        return
    rows_by_case = {row.get("case", ""): row for row in rows if row.get("suite", "") == "loombench"}
    for case, manifest_case in manifest_by_case.items():
        row = rows_by_case.get(case)
        if row is None:
            continue
        import_state = str(manifest_case.get("import_state", ""))
        manifest_app_case = str(manifest_case.get("manifest_case", ""))
        if import_state == "excluded":
            if row.get("status", "") != "unsupported":
                diagnostics.append(f"LoomBench excluded row {case} must stay unsupported")
            if row.get("diagnostic_class", "") != "loombench_import_excluded":
                diagnostics.append(f"LoomBench excluded row {case} has wrong diagnostic_class")
            if row_has_sim_artifacts(row):
                diagnostics.append(f"LoomBench excluded row {case} must not carry simulator artifacts")
        elif import_state == "deferred":
            if row.get("status", "") != "blocked":
                diagnostics.append(f"LoomBench deferred row {case} must stay blocked")
            if row.get("diagnostic_class", "") != "loombench_import_deferred":
                diagnostics.append(f"LoomBench deferred row {case} has wrong diagnostic_class")
            if row_has_sim_artifacts(row):
                diagnostics.append(f"LoomBench deferred row {case} must not carry simulator artifacts")
        elif import_state == "accepted":
            if row.get("status", "") == "pass":
                diagnostics.append(
                    f"LoomBench accepted row {case} cannot pass without explicit workload identity bridge"
                )
            if manifest_app_case != case:
                if row.get("diagnostic_class", "") != "loombench_workload_identity_bridge_missing":
                    diagnostics.append(f"LoomBench accepted alias row {case} must block on identity bridge")
            elif row.get("diagnostic_class", "") != "loombench_workload_identity_fingerprint_missing":
                diagnostics.append(f"LoomBench accepted row {case} must block on fingerprint bridge")
            if row_has_sim_artifacts(row):
                diagnostics.append(f"LoomBench accepted row {case} must not reuse simulator artifacts by name alone")


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
    if args.no_legacy_loombench and args.loombench_manifest:
        raise SystemExit("--no-legacy-loombench cannot be combined with --loombench-manifest")
    diagnostics: list[str] = []
    csv_input = Path(args.input)
    rows = read_csv_rows(csv_input, diagnostics)
    expected = expected_rows(
        Path(args.legacy_loombench_root),
        Path(args.loombench_manifest) if args.loombench_manifest else None,
        not args.no_legacy_loombench,
    )
    validate_rows(csv_input, rows, diagnostics)
    validate_coverage(rows, expected, diagnostics)
    validate_loombench_manifest_semantics(
        rows,
        Path(args.loombench_manifest) if args.loombench_manifest else None,
        diagnostics,
    )
    validate_json(json_path_for(csv_input, args.json_input), csv_input, rows, diagnostics)

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

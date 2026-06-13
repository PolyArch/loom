#!/usr/bin/env python3
"""Emit row-complete CGRA status baseline evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


STATUS_KEYS = ("pass", "fail", "blocked", "unsupported")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    parser.add_argument(
        "--legacy-loombench-root",
        default=str(ROOT / "temp" / "old_implementation_loom" / "loom" / "tests" / "app"),
    )
    return parser.parse_args(argv)


def empty_stage_fields() -> dict[str, str]:
    return {
        "hardware_system": "",
        "spatialcore_template": "",
        "mapping_id": "",
        "dfg_report": "",
        "dfg_report_fingerprint": "",
        "dfg_status": "not_run",
        "mapping_artifact": "",
        "mapping_artifact_fingerprint": "",
        "mapping_status": "not_run",
        "cgra_report": "",
        "cgra_report_fingerprint": "",
        "cgra_status": "not_run",
        "comparison_report": "",
        "comparison_report_fingerprint": "",
        "comparison_status": "not_run",
        "final_outputs_present": "false",
        "final_memory_state_present": "false",
        "status": "not_run",
        "diagnostic_class": "missing_status",
        "owner": "implementation",
    }


def row(
    *,
    suite: str,
    case: str,
    source_row: str,
    software_root: str,
    graph_ids: str = "",
    required_slice_count: str = "0",
    blocking_prerequisite: str,
    diagnostic: str,
) -> dict[str, str]:
    data = {
        "suite": suite,
        "case": case,
        "source_row": source_row,
        "software_root": software_root,
        "graph_ids": graph_ids,
        "required_slice_count": required_slice_count,
        "blocking_prerequisite": blocking_prerequisite,
        "diagnostic": diagnostic,
    }
    data.update(empty_stage_fields())
    return data


def load_app_manifest() -> dict[str, object]:
    path = ROOT / "test" / "app" / "manifest.json"
    return json.loads(path.read_text())


def app_rows() -> list[dict[str, str]]:
    manifest = load_app_manifest()
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("test/app/manifest.json cases must be a list")
    rows: list[dict[str, str]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case = str(entry.get("case", ""))
        if not case:
            continue
        tiers = entry.get("tiers", [])
        has_dfg = isinstance(tiers, list) and "dfg" in tiers
        prerequisite = "mapping_artifact" if has_dfg else "dataflow"
        diagnostic = (
            "CGRA status missing after app dataflow tier; mapping artifact and CGRA-sim report are absent"
            if has_dfg
            else "CGRA status missing because app row has no dataflow tier yet"
        )
        rows.append(
            row(
                suite="app",
                case=case,
                source_row=case,
                software_root=f"test/app/{case}",
                required_slice_count="1" if has_dfg else "0",
                blocking_prerequisite=prerequisite,
                diagnostic=diagnostic,
            )
        )
    return rows


def iter_target_rows(path: Path) -> Iterable[list[str]]:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield line.split("|")


def positive_shape(columns: list[str]) -> bool:
    if len(columns) < 17:
        return False
    # expect_thread through expect_demux are columns 6..15.
    for cell in columns[6:16]:
        text = cell.strip()
        if text.startswith(">="):
            text = text[2:]
        try:
            if int(text) > 0:
                return True
        except ValueError:
            continue
    return False


def cmsis_rows(suite: str, directory: str, targets_name: str, software_root: str) -> list[dict[str, str]]:
    targets = ROOT / "test" / directory / targets_name
    rows: list[dict[str, str]] = []
    for columns in iter_target_rows(targets):
        source = columns[0]
        has_shape = positive_shape(columns)
        prerequisite = "mapping_artifact" if has_shape else "dataflow_graph"
        diagnostic = (
            "CGRA status missing after CMSIS dataflow-shape row; mapping artifact and CGRA-sim report are absent"
            if has_shape
            else "CGRA status missing because CMSIS row emits no dataflow graph/thread shape"
        )
        rows.append(
            row(
                suite=suite,
                case=source,
                source_row=source,
                software_root=software_root,
                required_slice_count="1" if has_shape else "0",
                blocking_prerequisite=prerequisite,
                diagnostic=diagnostic,
            )
        )
    return rows


def loombench_rows(source_root: Path) -> list[dict[str, str]]:
    if not source_root.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        case = case_dir.name
        rows.append(
            row(
                suite="loombench",
                case=case,
                source_row=case,
                software_root=case_dir.relative_to(ROOT).as_posix()
                if case_dir.is_relative_to(ROOT)
                else case_dir.as_posix(),
                blocking_prerequisite="loombench_manifest",
                diagnostic="CGRA status missing because dedicated LoomBench manifest reconciliation is absent",
            )
        )
    return rows


def suite_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row_data in rows:
        suite = row_data["suite"]
        suite_counts = counts.setdefault(
            suite,
            {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "blocked": 0,
                "unsupported": 0,
                "missing_status": 0,
            },
        )
        suite_counts["total"] += 1
        status = row_data["status"]
        if status in STATUS_KEYS:
            suite_counts[status] += 1
        if row_data.get("diagnostic_class") == "missing_status":
            suite_counts["missing_status"] += 1
    return counts


def json_path_for(csv_output: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return csv_output.with_suffix(".json")


def write_json(path: Path, csv_output: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "kind": "cgra_status_summary",
        "csv_projection": str(csv_output),
        "counts": suite_counts(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    legacy_root = Path(args.legacy_loombench_root)
    rows = []
    rows.extend(app_rows())
    rows.extend(
        cmsis_rows(
            "cmsis-dsp",
            "cmsis-dsp",
            "cmsis_dsp_targets.txt",
            "externals/cmsis-dsp/Source",
        )
    )
    rows.extend(
        cmsis_rows(
            "cmsis-nn",
            "cmsis-nn",
            "cmsis_nn_targets.txt",
            "externals/cmsis-nn/Source",
        )
    )
    rows.extend(loombench_rows(legacy_root))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("cgra_status", output, rows)
    write_json(json_path_for(output, args.json_output), output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

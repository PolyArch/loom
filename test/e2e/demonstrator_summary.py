#!/usr/bin/env python3
"""Emit end-to-end demonstrator summary rows from intermediate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


CMSIS_SUITES = {"CMSIS-DSP", "CMSIS-NN"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def artifacts_by_kind(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def status_priority(status: str) -> int:
    order = {
        "fail": 0,
        "blocked": 1,
        "unsupported": 2,
        "skipped": 3,
        "not_run": 4,
        "pass": 5,
    }
    return order.get(status, 1)


def aggregate_statuses(statuses: list[str]) -> str:
    values = [status for status in statuses if status]
    if not values:
        return "blocked"
    return min(values, key=status_priority)


def source_compat_status(source_paths: list[Path], workload: str) -> str:
    statuses: list[str] = []
    for path in source_paths:
        for row in read_csv(path):
            if row.get("case") != workload:
                continue
            statuses.extend([row.get("native_status", ""), row.get("loom_status", "")])
    return aggregate_statuses(statuses)


def manifest_status(manifest_paths: list[Path]) -> str:
    if not manifest_paths:
        return "blocked"
    statuses: list[str] = []
    for path in manifest_paths:
        data = read_json(path)
        diagnostics = data.get("diagnostics", [])
        artifacts = data.get("artifacts", [])
        if diagnostics:
            statuses.append("blocked")
        elif artifacts:
            statuses.append("pass")
        else:
            statuses.append("blocked")
    return aggregate_statuses(statuses)


def manifest_artifact_ids(manifest_paths: list[Path]) -> set[str]:
    identities: set[str] = set()
    for path in manifest_paths:
        data = read_json(path)
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            identity = artifact.get("id")
            if isinstance(identity, str) and identity:
                identities.add(identity)
    return identities


def report_bundle_is_registered(grouped: dict[str, list[Path]], path: Path) -> bool:
    manifest_paths = grouped.get("artifact_manifest", [])
    if not manifest_paths:
        return False
    return intermediate_artifacts.artifact_id_for_path(path) in manifest_artifact_ids(manifest_paths)


def sim_status(sim_paths: list[Path], workload: str) -> str:
    statuses: list[str] = []
    for path in sim_paths:
        for row in read_csv(path):
            if row.get("kernel") == workload:
                status = row.get("status", "")
                if status:
                    statuses.append(status)
    return aggregate_statuses(statuses)


def hardware_matches(row_hardware: str, hardware: str) -> bool:
    return row_hardware == hardware or row_hardware.rsplit("::", 1)[-1] == hardware


def matching_rtl_fpa_rows(
    rtl_paths: list[Path],
    workload: str,
    hardware: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in rtl_paths:
        for row in read_csv(path):
            if row.get("workload") != workload:
                continue
            if hardware_matches(row.get("hardware", ""), hardware):
                rows.append(row)
    exact = [row for row in rows if row.get("hardware") == hardware]
    if exact:
        return exact
    return rows if len(rows) == 1 else []


def rtl_status(rtl_paths: list[Path], workload: str, hardware: str) -> str:
    statuses: list[str] = []
    for row in matching_rtl_fpa_rows(rtl_paths, workload, hardware):
        statuses.extend([row.get("rtl_lint_status", ""), row.get("rtl_sim_status", "")])
    return aggregate_statuses(statuses)


def fpa_status(rtl_paths: list[Path], workload: str, hardware: str) -> str:
    statuses: list[str] = []
    for row in matching_rtl_fpa_rows(rtl_paths, workload, hardware):
        status = row.get("status", "")
        if status:
            statuses.append(status)
    return aggregate_statuses(statuses)


def matching_report_bundle(
    report_paths: list[Path],
    workload: str,
    hardware: str,
    mapping_id: str,
) -> tuple[Path, dict[str, object]] | None:
    expected_suffix = f"::{mapping_id}" if mapping_id else ""
    for path in report_paths:
        data = read_json(path)
        if data.get("kind") != "workload_report_bundle":
            continue
        if data.get("workload") != workload:
            continue
        if data.get("selected_hardware_candidate_identity") != hardware:
            continue
        bundle_id = data.get("bundle_id", "")
        if expected_suffix and (not isinstance(bundle_id, str) or not bundle_id.endswith(expected_suffix)):
            continue
        return path, data
    return None


def report_bundle_status(
    grouped: dict[str, list[Path]],
    workload: str,
    hardware: str,
    mapping_id: str,
) -> tuple[str, str]:
    match = matching_report_bundle(grouped.get("workload_report_bundle", []), workload, hardware, mapping_id)
    if match is None:
        return "blocked", "workload report bundle is not available yet"
    path, bundle = match
    if not report_bundle_is_registered(grouped, path):
        return "blocked", "workload report bundle is absent from the artifact manifest"
    status = bundle.get("report_status", "blocked")
    if not isinstance(status, str) or not status:
        return "blocked", "workload report bundle has no report status"
    if status == "pass":
        return "pass", "workload report bundle available"
    diagnostics = bundle.get("diagnostics", [])
    if isinstance(diagnostics, list) and diagnostics:
        return status, "; ".join(str(item) for item in diagnostics)
    return status, f"workload report bundle status is {status}"


def matching_hardware_report_bundle(
    report_paths: list[Path],
    hardware: str,
) -> tuple[Path, dict[str, object]] | None:
    for path in report_paths:
        data = read_json(path)
        if data.get("kind") != "hardware_report_bundle":
            continue
        if data.get("hardware_candidate_identity") != hardware:
            continue
        return path, data
    return None


def hardware_report_status(grouped: dict[str, list[Path]], hardware: str) -> tuple[str, str]:
    match = matching_hardware_report_bundle(grouped.get("hardware_report_bundle", []), hardware)
    if match is None:
        return "blocked", "hardware candidate verified; hardware-only report bundle is not available yet"
    path, bundle = match
    if not report_bundle_is_registered(grouped, path):
        return "blocked", "hardware report bundle is absent from the artifact manifest"
    status = bundle.get("report_status", "blocked")
    if not isinstance(status, str) or not status:
        return "blocked", "hardware report bundle has no report status"
    if status == "pass":
        return "pass", "hardware report bundle available"
    diagnostics = bundle.get("diagnostics", [])
    if isinstance(diagnostics, list) and diagnostics:
        return status, "; ".join(str(item) for item in diagnostics)
    return status, f"hardware report bundle status is {status}"


def mapping_rows(mapping_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in mapping_paths:
        rows.extend(
            row
            for row in read_csv(path)
            if row.get("workload") not in {"", "scaffold", None}
            and row.get("hardware") not in {"", "scaffold", None}
        )
    return rows


def demonstrator_row(grouped: dict[str, list[Path]], row: dict[str, str]) -> dict[str, str]:
    workload = row["workload"]
    hardware = row["hardware"]
    mapping_id = row.get("mapping_id", "")
    report_status, diagnostic = report_bundle_status(grouped, workload, hardware, mapping_id)
    return {
        "demonstrator": f"app::{workload}::{hardware}",
        "compat_status": source_compat_status(grouped.get("source_compat", []), workload),
        "artifact_status": manifest_status(grouped.get("artifact_manifest", [])),
        "mapping_status": row.get("status", "blocked"),
        "sim_status": sim_status(grouped.get("sim_cycle", []), workload),
        "rtl_status": rtl_status(grouped.get("rtl_fpa", []), workload, hardware),
        "fpa_status": fpa_status(grouped.get("rtl_fpa", []), workload, hardware),
        "report_status": report_status,
        "diagnostic": diagnostic,
    }


def cmsis_demonstrator_rows(grouped: dict[str, list[Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in grouped.get("compiler_pipeline", []):
        for row in read_csv(path):
            suite = row.get("suite", "")
            case = row.get("case", "")
            if suite not in CMSIS_SUITES or not case:
                continue
            artifact_status = aggregate_statuses(
                [
                    row.get("llvm_ir_status", ""),
                    row.get("raised_mlir_status", ""),
                    row.get("dataflow_status", ""),
                ]
            )
            compat_status = aggregate_statuses([row.get("llvm_ir_status", "")])
            downstream_status = "skipped" if artifact_status == "pass" else "blocked"
            rows.append(
                {
                    "demonstrator": f"cmsis::{case}",
                    "compat_status": compat_status,
                    "artifact_status": artifact_status,
                    "mapping_status": downstream_status,
                    "sim_status": downstream_status,
                    "rtl_status": downstream_status,
                    "fpa_status": downstream_status,
                    "report_status": downstream_status,
                    "diagnostic": (
                        "CMSIS drop-in pipeline reached dataflow; mapped reports "
                        "require a compatible ADG profile"
                        if artifact_status == "pass"
                        else row.get("diagnostic", "CMSIS compiler pipeline is incomplete")
                    ),
                }
            )
    return rows


def hardware_demonstrator_rows(grouped: dict[str, list[Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in grouped.get("adg_hardware", []):
        for row in read_csv(path):
            hardware = row.get("hardware", "")
            if not hardware or hardware == "scaffold":
                continue
            verify_status = row.get("verify_status", "blocked")
            report_status, diagnostic = hardware_report_status(grouped, hardware)
            rows.append(
                {
                    "demonstrator": f"hardware::{hardware}",
                    "compat_status": "skipped",
                    "artifact_status": verify_status,
                    "mapping_status": "skipped",
                    "sim_status": "skipped",
                    "rtl_status": "skipped",
                    "fpa_status": "skipped",
                    "report_status": report_status if verify_status == "pass" else "blocked",
                    "diagnostic": diagnostic if verify_status == "pass" else row.get("diagnostic", "hardware candidate did not verify"),
                }
            )
    return rows


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = intermediate_artifacts.discover_artifact_paths(
        ROOT,
        args.artifact,
        include_unsupported_scope=False,
    )
    grouped = artifacts_by_kind(paths)
    rows = [demonstrator_row(grouped, row) for row in mapping_rows(grouped.get("pnr_mapping", []))]
    rows.extend(cmsis_demonstrator_rows(grouped))
    rows.extend(hardware_demonstrator_rows(grouped))

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("e2e_demonstrator", output, rows)
    else:
        intermediate_artifacts.write_csv("e2e_demonstrator", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

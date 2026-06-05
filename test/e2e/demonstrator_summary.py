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


def sim_status(sim_paths: list[Path], workload: str) -> str:
    statuses: list[str] = []
    for path in sim_paths:
        for row in read_csv(path):
            if row.get("kernel") == workload:
                status = row.get("status", "")
                if status:
                    statuses.append(status)
                elif row.get("dfg_sim_cycles") and row.get("cgra_sim_cycles"):
                    statuses.append("pass")
    return aggregate_statuses(statuses)


def rtl_status(rtl_paths: list[Path], workload: str, hardware: str) -> str:
    statuses: list[str] = []
    for path in rtl_paths:
        for row in read_csv(path):
            if row.get("workload") == workload and row.get("hardware") == hardware:
                statuses.extend([row.get("rtl_lint_status", ""), row.get("rtl_sim_status", "")])
    return aggregate_statuses(statuses)


def fpa_status(rtl_paths: list[Path], workload: str, hardware: str) -> str:
    statuses: list[str] = []
    for path in rtl_paths:
        for row in read_csv(path):
            if row.get("workload") == workload and row.get("hardware") == hardware:
                statuses.extend([row.get("synth_status", ""), row.get("status", "")])
    return aggregate_statuses(statuses)


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
    return {
        "demonstrator": f"app::{workload}::{hardware}",
        "compat_status": source_compat_status(grouped.get("source_compat", []), workload),
        "artifact_status": manifest_status(grouped.get("artifact_manifest", [])),
        "mapping_status": row.get("status", "blocked"),
        "sim_status": sim_status(grouped.get("sim_cycle", []), workload),
        "rtl_status": rtl_status(grouped.get("rtl_fpa", []), workload, hardware),
        "fpa_status": fpa_status(grouped.get("rtl_fpa", []), workload, hardware),
        "report_status": "blocked",
        "diagnostic": "workload report bundle is not available yet",
    }


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

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("e2e_demonstrator", output, rows)
    else:
        intermediate_artifacts.write_csv("e2e_demonstrator", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

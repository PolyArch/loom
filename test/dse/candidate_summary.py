#!/usr/bin/env python3
"""Emit DSE candidate summary rows from mapping, sim, and FPA artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--objective", default="minimize_runtime")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def artifacts_by_kind(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def mapping_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(
            row
            for row in read_csv(path)
            if row.get("workload") not in {"", "scaffold", None}
            and row.get("hardware") not in {"", "scaffold", None}
        )
    return rows


def sim_for_workload(paths: list[Path], workload: str) -> dict[str, str]:
    for path in paths:
        for row in read_csv(path):
            if row.get("kernel") == workload:
                return row
    return {}


def fpa_for_candidate(paths: list[Path], workload: str, hardware: str) -> dict[str, str]:
    for path in paths:
        for row in read_csv(path):
            if row.get("workload") == workload and row.get("hardware") == hardware:
                return row
    return {}


def candidate_row(
    mapping: dict[str, str],
    sim: dict[str, str],
    fpa: dict[str, str],
    objective: str,
) -> dict[str, str]:
    workload = mapping["workload"]
    hardware = mapping["hardware"]
    _ = (sim, fpa)
    return {
        "candidate": f"candidate::{workload}::{hardware}",
        "workload": workload,
        "hardware": hardware,
        "mapping_id": "",
        "objective": objective,
        "cgra_sim_cycles": "",
        "frequency_mhz": "",
        "area_um2": "",
        "dynamic_power_mw": "",
        "energy_nj": "",
        "selection_status": "blocked",
        "diagnostic": "missing mapping, simulator, or FPA evidence for DSE selection",
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
    rows = [
        candidate_row(
            row,
            sim_for_workload(grouped.get("sim_cycle", []), row["workload"]),
            fpa_for_candidate(grouped.get("rtl_fpa", []), row["workload"], row["hardware"]),
            args.objective,
        )
        for row in mapping_rows(grouped.get("pnr_mapping", []))
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("dse_candidate", output, rows)
    else:
        intermediate_artifacts.write_csv("dse_candidate", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Emit PnR mapping summary rows from software and hardware summaries."""

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
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def workloads_from_primitive_coverage(path: Path) -> list[str]:
    workloads = {
        row["workload"]
        for row in read_rows(path)
        if row.get("workload") not in {"", "scaffold", "none", None}
    }
    return sorted(workloads)


def hardware_from_summary(path: Path) -> list[str]:
    hardware = {
        row["hardware"]
        for row in read_rows(path)
        if row.get("verify_status") == "pass" and row.get("hardware")
    }
    return sorted(hardware)


def mapping_row(workload: str, hardware: str) -> dict[str, str]:
    return {
        "workload": workload,
        "hardware": hardware,
        "mapping_id": "",
        "placed_records": "",
        "routed_edges": "",
        "unrouted_edges": "",
        "status": "blocked",
        "diagnostic": "PnR mapping artifact producer is not implemented yet; software and hardware candidates were discovered",
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    primitive_path = Path(args.primitive_coverage) if args.primitive_coverage else ROOT / "temp/dataflow-primitive-coverage.csv"
    hardware_path = Path(args.hardware_summary) if args.hardware_summary else ROOT / "temp/adg-hardware-summary.csv"

    workloads = workloads_from_primitive_coverage(primitive_path)
    hardware = hardware_from_summary(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("pnr_mapping", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [mapping_row(workload, candidate) for workload in workloads for candidate in hardware]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("pnr_mapping", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

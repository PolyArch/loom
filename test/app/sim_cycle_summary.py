#!/usr/bin/env python3
"""Emit simulator cycle summary rows."""

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
    return parser.parse_args(argv)


def workloads_from_primitive_coverage(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return sorted({row["workload"] for row in csv.DictReader(handle) if row.get("workload")})


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    primitive_path = Path(args.primitive_coverage) if args.primitive_coverage else ROOT / "temp/dataflow-primitive-coverage.csv"
    if not primitive_path.is_file():
        intermediate_artifacts.write_csv("sim_cycle", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [
        {
            "kernel": workload,
            "dfg_sim_cycles": "",
            "cgra_sim_cycles": "",
            "status": "blocked",
            "diagnostic": "DFG-sim and CGRA-sim cycle evidence is not available yet",
        }
        for workload in workloads_from_primitive_coverage(primitive_path)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("sim_cycle", output, rows)
    else:
        intermediate_artifacts.write_csv("sim_cycle", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

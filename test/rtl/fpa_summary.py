#!/usr/bin/env python3
"""Emit RTL/FPA summary rows from software and hardware summaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import candidate_summary_common  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    return parser.parse_args(argv)


def fpa_row(workload: str, hardware: str) -> dict[str, str]:
    return {
        "hardware": hardware,
        "workload": workload,
        "rtl_lint_status": "blocked",
        "rtl_sim_status": "blocked",
        "synth_status": "blocked",
        "frequency_mhz": "",
        "area_um2": "",
        "dynamic_power_mw": "",
        "leakage_power_mw": "",
        "status": "blocked",
        "diagnostic": "RTL/FPA backend evidence is not available yet",
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    ignore_standard = os.environ.get("LOOM_IGNORE_STANDARD_ARTIFACTS") == "1"
    primitive_path = (
        Path(args.primitive_coverage)
        if args.primitive_coverage
        else Path()
        if ignore_standard
        else ROOT / "temp/dataflow-primitive-coverage.csv"
    )
    hardware_path = (
        Path(args.hardware_summary)
        if args.hardware_summary
        else Path()
        if ignore_standard
        else ROOT / "temp/adg-hardware-summary.csv"
    )

    workloads = candidate_summary_common.workloads_from_primitive_coverage(primitive_path)
    hardware = candidate_summary_common.hardware_from_summary(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("rtl_fpa", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [fpa_row(workload, candidate) for workload in workloads for candidate in hardware]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("rtl_fpa", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

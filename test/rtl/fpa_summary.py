#!/usr/bin/env python3
"""Emit RTL/FPA summary rows from software and hardware summaries."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import candidate_summary_common  # noqa: E402


@dataclass(frozen=True)
class HardwareCandidate:
    name: str
    node_count: int
    link_count: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    return parser.parse_args(argv)


def read_hardware_candidates(path: Path) -> list[HardwareCandidate]:
    if not path.is_file():
        return []
    candidates: list[HardwareCandidate] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verify_status") != "pass" or not row.get("hardware"):
                continue
            try:
                node_count = int(row.get("node_count", ""))
                link_count = int(row.get("link_count", ""))
            except ValueError:
                continue
            if node_count <= 0 or link_count < 0:
                continue
            candidates.append(
                HardwareCandidate(
                    name=row["hardware"],
                    node_count=node_count,
                    link_count=link_count,
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.name)


def format_estimate(value: float) -> str:
    return f"{value:.3f}"


def analytic_fpa_row(workload: str, hardware: HardwareCandidate) -> dict[str, str]:
    node_count = hardware.node_count
    link_count = hardware.link_count
    area_um2 = 1000.0 + 250.0 * node_count + 50.0 * link_count
    frequency_mhz = max(50.0, 500.0 - 10.0 * node_count - 5.0 * link_count)
    dynamic_power_mw = 1.0 + 0.2 * node_count + 0.05 * link_count
    leakage_power_mw = 0.1 + 0.0001 * area_um2
    return {
        "hardware": hardware.name,
        "workload": workload,
        "rtl_lint_status": "skipped",
        "rtl_sim_status": "skipped",
        "synth_status": "skipped",
        "frequency_mhz": format_estimate(frequency_mhz),
        "area_um2": format_estimate(area_um2),
        "dynamic_power_mw": format_estimate(dynamic_power_mw),
        "leakage_power_mw": format_estimate(leakage_power_mw),
        "status": "pass",
        "diagnostic": (
            "analytic FPA estimate; fidelity=analytic; "
            "activity_source=default_toggle; RTL backend not run"
        ),
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
    hardware = read_hardware_candidates(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("rtl_fpa", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [
        analytic_fpa_row(workload, candidate)
        for workload in workloads
        for candidate in hardware
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("rtl_fpa", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

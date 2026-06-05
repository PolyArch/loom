#!/usr/bin/env python3
"""Emit PnR mapping summary rows from software and hardware summaries."""

from __future__ import annotations

import argparse
import os
import subprocess
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
    parser.add_argument("--dfg-mlir")
    parser.add_argument("--graph")
    parser.add_argument("--hardware-mlir")
    parser.add_argument("--hardware")
    parser.add_argument("--workload")
    parser.add_argument("--artifact")
    return parser.parse_args(argv)


def mapping_row(workload: str, hardware: str) -> dict[str, str]:
    return {
        "workload": workload,
        "hardware": hardware,
        "mapping_id": "",
        "placed_records": "",
        "routed_edges": "",
        "unrouted_edges": "",
        "unplaced_records": "",
        "status": "blocked",
        "diagnostic": "PnR mapping artifact producer is not implemented yet; software and hardware candidates were discovered",
    }


def tool_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_PNR_MAP")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-pnr-map/loom-pnr-map",
            ROOT / "build/bin/loom-pnr-map",
        ]
    )
    return candidates


def find_tool() -> Path | None:
    for candidate in tool_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def explicit_mapper_args(args: argparse.Namespace) -> bool:
    explicit = [
        args.dfg_mlir,
        args.graph,
        args.hardware_mlir,
        args.hardware,
        args.workload,
    ]
    if any(explicit) and not all(explicit):
        missing = [
            name
            for name, value in (
                ("--dfg-mlir", args.dfg_mlir),
                ("--graph", args.graph),
                ("--hardware-mlir", args.hardware_mlir),
                ("--hardware", args.hardware),
                ("--workload", args.workload),
            )
            if not value
        ]
        raise SystemExit(f"explicit mapper mode is missing {', '.join(missing)}")
    return all(explicit)


def run_explicit_mapper(args: argparse.Namespace) -> int:
    tool = find_tool()
    if tool is None:
        sys.stderr.write("missing loom-pnr-map; build the mapper tool first\n")
        return 1
    command = [
        str(tool),
        "--dfg-mlir",
        str(Path(args.dfg_mlir)),
        "--graph",
        args.graph,
        "--hardware-mlir",
        str(Path(args.hardware_mlir)),
        "--hardware",
        args.hardware,
        "--workload",
        args.workload,
        "--output",
        args.output,
    ]
    if args.artifact:
        command.extend(["--artifact", args.artifact])
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result.returncode


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if explicit_mapper_args(args):
        return run_explicit_mapper(args)

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
        intermediate_artifacts.write_csv("pnr_mapping", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [mapping_row(workload, candidate) for workload in workloads for candidate in hardware]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("pnr_mapping", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

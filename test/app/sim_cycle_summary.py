#!/usr/bin/env python3
"""Emit simulator cycle summary rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--dfg-report", action="append", default=[])
    parser.add_argument("--cgra-report", action="append", default=[])
    return parser.parse_args(argv)


def workloads_from_primitive_coverage(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return sorted({row["workload"] for row in csv.DictReader(handle) if row.get("workload")})


def tool_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_SIM_CYCLE_SUMMARY")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-sim-cycle-summary/loom-sim-cycle-summary",
            ROOT / "build/bin/loom-sim-cycle-summary",
        ]
    )
    return candidates


def find_existing_tool(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_tool() -> Path | None:
    return find_existing_tool(tool_candidates())


def classify_report(path: Path) -> str:
    with path.open() as handle:
        data = json.load(handle)
    kind = data.get("kind")
    if kind not in {"dfg_sim_report", "cgra_sim_report"}:
        raise ValueError(f"{path} has unsupported simulator report kind {kind!r}")
    return kind


def discover_report_inputs(evidence_dir: Path) -> tuple[list[Path], list[Path]]:
    dfg_reports: list[Path] = []
    cgra_reports: list[Path] = []
    if not evidence_dir.is_dir():
        return dfg_reports, cgra_reports
    reports = sorted(
        {
            *evidence_dir.glob("*.report.json"),
            *evidence_dir.glob("*-dfg-sim-report.json"),
            *evidence_dir.glob("*-cgra-sim-report.json"),
        }
    )
    for report in reports:
        kind = classify_report(report)
        if kind == "dfg_sim_report":
            dfg_reports.append(report)
        else:
            cgra_reports.append(report)
    return dfg_reports, cgra_reports


def dfg_sim_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_DFG_SIM")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-dfg-sim/loom-dfg-sim",
            ROOT / "build/bin/loom-dfg-sim",
        ]
    )
    return candidates


def cgra_sim_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_CGRA_SIM")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-cgra-sim/loom-cgra-sim",
            ROOT / "build/bin/loom-cgra-sim",
        ]
    )
    return candidates


def write_blocked_default(output: Path, diagnostic: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows(
        "sim_cycle",
        output,
        [
            {
                "kernel": "vecsum",
                "dfg_sim_cycles": "",
                "cgra_sim_cycles": "",
                "status": "blocked",
                "diagnostic": diagnostic,
            }
        ],
    )


def run_command(command: list[str], env: dict[str, str] | None = None) -> int:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result.returncode


def summarize_reports(output: Path, dfg_reports: list[Path], cgra_reports: list[Path]) -> int:
    tool = find_tool()
    if tool is None:
        write_blocked_default(output, "missing loom-sim-cycle-summary tool")
        return 0
    command = [str(tool)]
    for report in dfg_reports:
        command.extend(["--dfg-report", str(report)])
    for report in cgra_reports:
        command.extend(["--cgra-report", str(report)])
    command.extend(["--output", str(output)])
    return run_command(command)


def emit_default_vecsum_summary(output: Path) -> int:
    dfg_tool = find_existing_tool(dfg_sim_candidates())
    cgra_tool = find_existing_tool(cgra_sim_candidates())
    if dfg_tool is None:
        write_blocked_default(output, "missing loom-dfg-sim tool for default vecsum simulator evidence")
        return 0
    if cgra_tool is None:
        write_blocked_default(output, "missing loom-cgra-sim tool for default vecsum simulator evidence")
        return 0

    work_dir = output.parent / f"{output.stem}-default-evidence"
    dfg_dir = work_dir / "vecsum-dfg"
    dfg_report = output.parent / "vecsum-dfg-sim-report.json"
    dfg_cycle = work_dir / "vecsum-dfg-sim-cycle-summary.csv"
    mapping_summary = work_dir / "pnr-mapping-summary.csv"
    mapping_artifact = output.parent / "pnr-mapping.json"
    cgra_report = output.parent / "vecsum-cgra-sim-report.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "BUILD_DIR": str(dfg_dir),
            "LOOM_CC": str(ROOT / "build/bin/loom-cc"),
            "LOOM_RAISE": str(ROOT / "build/bin/loom-raise"),
            "LOOM_LOWER": str(ROOT / "build/bin/loom-lower"),
            "LOOM_RAISE_OPT": str(ROOT / "build/bin/loom-raise-opt"),
        }
    )
    dfg_check = ROOT / "test/app/vecsum/dfg_check.sh"
    status = run_command(["bash", str(dfg_check)], env=env)
    if status != 0:
        return status

    dfg_mlir = dfg_dir / "main_func.dfg.mlir"
    env = os.environ.copy()
    env["LOOM_DFG_SIM"] = str(dfg_tool)
    status = run_command(
        [
            "bash",
            str(ROOT / "test/simulator/run_app_reduction_dfg_sim.sh"),
            "vecsum",
            str(dfg_mlir),
            str(dfg_report),
            str(dfg_cycle),
        ],
        env=env,
    )
    if status != 0:
        return status

    status = run_command(
        [
            "bash",
            str(ROOT / "test/pnr/run_mapping_summary.sh"),
            "--dfg-mlir",
            str(dfg_mlir),
            "--graph",
            "g_t_vecsum_red_0_0",
            "--hardware-mlir",
            str(ROOT / "test/pnr/shared_reduction_adg.mlir"),
            "--hardware",
            "shared_reduction_adg",
            "--workload",
            "vecsum",
            "--artifact",
            str(mapping_artifact),
            "--output",
            str(mapping_summary),
        ]
    )
    if status != 0:
        return status

    status = run_command(
        [
            str(cgra_tool),
            "--dfg-report",
            str(dfg_report),
            "--mapping-artifact",
            str(mapping_artifact),
            "--hardware-mlir",
            str(ROOT / "test/pnr/shared_reduction_adg.mlir"),
            "--output",
            str(cgra_report),
        ]
    )
    if status != 0:
        return status

    return summarize_reports(output, [dfg_report], [cgra_report])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    dfg_reports = [Path(path) for path in args.dfg_report]
    valid_dfg_reports = [path for path in dfg_reports if path.is_file()]
    cgra_reports = [Path(path) for path in args.cgra_report]
    valid_cgra_reports = [path for path in cgra_reports if path.is_file()]
    if not args.primitive_coverage and not valid_dfg_reports:
        discovered_dfg_reports, discovered_cgra_reports = discover_report_inputs(
            output.parent / "current-sim-cycle"
        )
        if discovered_dfg_reports:
            tool = find_tool()
            if tool is not None:
                return summarize_reports(output, discovered_dfg_reports, discovered_cgra_reports)
            intermediate_artifacts.write_csv("sim_cycle", output)
            return 0
        return emit_default_vecsum_summary(output)
    if valid_dfg_reports:
        tool = find_tool()
        if tool is not None:
            return summarize_reports(output, valid_dfg_reports, valid_cgra_reports)
        intermediate_artifacts.write_csv("sim_cycle", output)
        return 0
    primitive_path = Path(args.primitive_coverage)
    if not primitive_path.is_file():
        intermediate_artifacts.write_csv("sim_cycle", intermediate_artifacts.output_path(args.output))
        return 0

    tool = find_tool()
    if tool is not None:
        result = subprocess.run(
            [
                str(tool),
                "--primitive-coverage",
                str(primitive_path),
                "--output",
                str(output),
            ],
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

#!/usr/bin/env python3
"""Run bounded CMSIS DFG-sim attempts for row-level status evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


@dataclass(frozen=True)
class Attempt:
    suite: str
    case: str
    stem: str
    graph: str
    dfg_dir_arg: str
    args: tuple[str, ...]
    memrefs: tuple[str, ...]


ATTEMPTS = (
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_abs_f32.c",
        stem="arm_abs_f32",
        graph="g_t_arm_abs_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=-1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmsis-dsp-dfg-dir", required=True)
    parser.add_argument("--cmsis-nn-dfg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loom-dfg-sim")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def resolve_tool(explicit: str | None) -> Path:
    value = explicit or os.environ.get("LOOM_DFG_SIM")
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate
        resolved = shutil.which(value)
        if resolved:
            return Path(resolved)
    return ROOT / "build" / "tools" / "loom-dfg-sim" / "loom-dfg-sim"


def run_attempt(tool: Path, output_dir: Path, args: argparse.Namespace, attempt: Attempt) -> Path:
    dfg_dir = Path(getattr(args, attempt.dfg_dir_arg))
    dfg_mlir = dfg_dir / f"{attempt.stem}.dfg.mlir"
    if not dfg_mlir.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] missing DFG MLIR for {attempt.case}: {dfg_mlir}")

    output = output_dir / f"{attempt.stem}.dfg.report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(tool),
        str(dfg_mlir),
        "--graph",
        attempt.graph,
        "--workload",
        attempt.case,
        "--output",
        str(output),
    ]
    for arg in attempt.args:
        command.extend(["--arg", arg])
    for memref in attempt.memrefs:
        command.extend(["--memref", memref])

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[cmsis-dfg-sim] {attempt.case} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not output.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} produced no report: {output}")
    data = json.loads(output.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} report is not a JSON object: {output}")
    data["input_artifact_fingerprints"] = intermediate_artifacts.input_artifact_fingerprints([dfg_mlir])
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tool = resolve_tool(args.loom_dfg_sim)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise SystemExit(f"[cmsis-dfg-sim] loom-dfg-sim not executable: {tool}")
    output_dir = Path(args.output_dir)
    reports = [run_attempt(tool, output_dir, args, attempt) for attempt in ATTEMPTS]
    for report in reports:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

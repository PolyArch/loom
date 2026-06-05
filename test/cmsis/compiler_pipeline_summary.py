#!/usr/bin/env python3
"""Emit CMSIS compiler pipeline summary rows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def repo_temp_dir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    temp_root = ROOT / "temp" / "test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)


def build_tool_path(env_name: str, fallback_name: str) -> str:
    return os.environ.get(env_name, str(ROOT / "build" / "bin" / fallback_name))


def run_script(script: Path, out_dir: Path) -> tuple[str, str]:
    env = os.environ.copy()
    env["LOOM_CC"] = build_tool_path("LOOM_CC", "loom-cc")
    env["LOOM_RAISE"] = build_tool_path("LOOM_RAISE", "loom-raise")
    env["LOOM_LOWER"] = build_tool_path("LOOM_LOWER", "loom-lower")
    env["LOOM_RAISE_OPT"] = build_tool_path("LOOM_RAISE_OPT", "loom-raise-opt")
    env["OUT_OVERRIDE"] = str(out_dir)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return "pass", result.stdout.strip()
    detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return "fail", detail[0] if detail else f"{script.name} exited {result.returncode}"


def run_suite(case: str, suite: str, directory: str, prefix: str) -> dict[str, str]:
    suite_dir = ROOT / "test" / directory
    with repo_temp_dir(f"loom-cmsis-{case}-") as tmp:
        temp_root = Path(tmp)
        ir_status, ir_diag = run_script(suite_dir / f"run_{prefix}_ir.sh", temp_root / "ir")
        if ir_status != "pass":
            return {
                "case": case,
                "suite": suite,
                "llvm_ir_status": "fail",
                "raised_mlir_status": "blocked",
                "dataflow_status": "blocked",
                "diagnostic": ir_diag,
            }
        raise_status, raise_diag = run_script(suite_dir / f"run_{prefix}_raise.sh", temp_root / "raise")
        if raise_status != "pass":
            return {
                "case": case,
                "suite": suite,
                "llvm_ir_status": "pass",
                "raised_mlir_status": "fail",
                "dataflow_status": "blocked",
                "diagnostic": raise_diag,
            }
        dfg_status, dfg_diag = run_script(suite_dir / f"run_{prefix}_dfg.sh", temp_root / "dfg")
        if dfg_status != "pass":
            return {
                "case": case,
                "suite": suite,
                "llvm_ir_status": "pass",
                "raised_mlir_status": "pass",
                "dataflow_status": "fail",
                "diagnostic": dfg_diag,
            }
    return {
        "case": case,
        "suite": suite,
        "llvm_ir_status": "pass",
        "raised_mlir_status": "pass",
        "dataflow_status": "pass",
        "diagnostic": "IR, raise, and dataflow checks passed",
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rows = [
        run_suite("cmsis-dsp", "CMSIS-DSP", "cmsis-dsp", "cmsis_dsp"),
        run_suite("cmsis-nn", "CMSIS-NN", "cmsis-nn", "cmsis_nn"),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("compiler_pipeline", output, rows)
    status_columns = ("llvm_ir_status", "raised_mlir_status", "dataflow_status")
    return 1 if any(row[column] == "fail" for row in rows for column in status_columns) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

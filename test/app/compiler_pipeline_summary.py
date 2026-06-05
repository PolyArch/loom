#!/usr/bin/env python3
"""Emit app compiler pipeline summary rows."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


HEADER = [
    "case",
    "suite",
    "llvm_ir_status",
    "raised_mlir_status",
    "dataflow_status",
    "diagnostic",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    return parser.parse_args(argv)


def discover_cases() -> list[str]:
    app_root = ROOT / "test" / "app"
    return sorted(
        path.name
        for path in app_root.iterdir()
        if (path / "raise_check.sh").is_file() and (path / "dfg_check.sh").is_file()
    )


def tool_path(env_name: str, fallback: str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    return str(ROOT / "build" / "bin" / fallback)


def prepare_temp_app(source_dir: Path, tmp: str) -> Path:
    tmp_root = Path(tmp)
    app_root = tmp_root / "test" / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "test" / "app" / "dfg_common.sh", app_root / "dfg_common.sh")
    work_dir = app_root / source_dir.name
    shutil.copytree(source_dir, work_dir, ignore=shutil.ignore_patterns("build"))
    return work_dir


def run_script(script: Path, env: dict[str, str]) -> tuple[str, str]:
    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parent,
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


def run_case(source_dir: Path) -> tuple[str, str, str, str]:
    with tempfile.TemporaryDirectory(prefix=f"loom-pipeline-{source_dir.name}-") as tmp:
        work_dir = prepare_temp_app(source_dir, tmp)
        env = os.environ.copy()
        env["LOOM_CC"] = tool_path("LOOM_CC", "loom-cc")
        env["LOOM_CXX"] = tool_path("LOOM_CXX", "loom-c++")
        env["LOOM_RAISE"] = tool_path("LOOM_RAISE", "loom-raise")
        env["LOOM_LOWER"] = tool_path("LOOM_LOWER", "loom-lower")
        env["LOOM_RAISE_OPT"] = tool_path("LOOM_RAISE_OPT", "loom-raise-opt")

        raise_status, raise_diag = run_script(work_dir / "raise_check.sh", env)
        if raise_status != "pass":
            return "fail", "fail", "blocked", raise_diag
        dfg_status, dfg_diag = run_script(work_dir / "dfg_check.sh", env)
        if dfg_status != "pass":
            return "pass", "pass", "fail", dfg_diag
    return "pass", "pass", "pass", "LLVM IR, raise, and dataflow checks passed"


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    cases = args.cases or discover_cases()
    if not cases:
        intermediate_artifacts.write_csv("compiler_pipeline", intermediate_artifacts.output_path(args.output))
        return 0

    rows: list[dict[str, str]] = []
    failed = False
    for case in cases:
        source_dir = ROOT / "test" / "app" / case
        if not (source_dir / "raise_check.sh").is_file() or not (source_dir / "dfg_check.sh").is_file():
            rows.append(
                {
                    "case": case,
                    "suite": "app",
                    "llvm_ir_status": "blocked",
                    "raised_mlir_status": "blocked",
                    "dataflow_status": "blocked",
                    "diagnostic": "missing app raise_check.sh or dfg_check.sh",
                }
            )
            failed = True
            continue
        llvm_status, raise_status, dfg_status, diagnostic = run_case(source_dir)
        if (llvm_status, raise_status, dfg_status) != ("pass", "pass", "pass"):
            failed = True
        rows.append(
            {
                "case": case,
                "suite": "app",
                "llvm_ir_status": llvm_status,
                "raised_mlir_status": raise_status,
                "dataflow_status": dfg_status,
                "diagnostic": diagnostic,
            }
        )

    write_rows(output, rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

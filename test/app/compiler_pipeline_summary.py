#!/usr/bin/env python3
"""Emit app compiler pipeline summary rows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "app"))

import app_summary_common  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    return parser.parse_args(argv)


def discover_cases() -> list[str]:
    raise_cases = set(app_summary_common.discover_app_cases("raise"))
    dfg_cases = set(app_summary_common.discover_app_cases("dfg"))
    return sorted(raise_cases & dfg_cases)


def tool_path(env_name: str, fallback: str) -> str:
    return app_summary_common.build_tool_path(env_name, fallback)


def run_script(script: Path, env: dict[str, str]) -> tuple[str, str]:
    return app_summary_common.run_bash_script(script, env)


def run_case(source_dir: Path) -> tuple[str, str, str, str]:
    with app_summary_common.repo_temp_dir(f"loom-pipeline-{source_dir.name}-") as tmp:
        env = os.environ.copy()
        env["LOOM_CC"] = tool_path("LOOM_CC", "loom-cc")
        env["LOOM_CXX"] = tool_path("LOOM_CXX", "loom-c++")
        env["LOOM_RAISE"] = tool_path("LOOM_RAISE", "loom-raise")
        env["LOOM_LOWER"] = tool_path("LOOM_LOWER", "loom-lower")
        env["LOOM_RAISE_OPT"] = tool_path("LOOM_RAISE_OPT", "loom-raise-opt")

        env["BUILD_DIR"] = str(Path(tmp) / "raise")
        raise_status, raise_diag = run_script(source_dir / "raise_check.sh", env)
        if raise_status != "pass":
            return "fail", "fail", "blocked", raise_diag
        env["BUILD_DIR"] = str(Path(tmp) / "dfg")
        dfg_status, dfg_diag = run_script(source_dir / "dfg_check.sh", env)
        if dfg_status != "pass":
            return "pass", "pass", "fail", dfg_diag
    return "pass", "pass", "pass", "LLVM IR, raise, and dataflow checks passed"


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    app_summary_common.write_rows("compiler_pipeline", output, rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    cases = args.cases or discover_cases()
    if not cases:
        app_summary_common.write_empty("compiler_pipeline", args.output)
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

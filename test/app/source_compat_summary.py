#!/usr/bin/env python3
"""Emit app source compatibility summary rows."""

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
    return app_summary_common.discover_app_cases("run")


def compiler_path(env_name: str, fallback: Path) -> str:
    return app_summary_common.env_path(env_name, fallback)


def run_case(source_dir: Path, cc: str, cxx: str, label: str) -> tuple[str, str]:
    with app_summary_common.repo_temp_dir(f"loom-app-{source_dir.name}-") as tmp:
        env = os.environ.copy()
        env["CC"] = cc
        env["CXX"] = cxx
        env["BUILD_DIR"] = str(Path(tmp) / label)
        return app_summary_common.run_bash_script(
            source_dir / "run_check.sh",
            env=env,
            cwd=ROOT,
        )


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    app_summary_common.write_rows("source_compat", output, rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    cases = args.cases or discover_cases()
    if not cases:
        app_summary_common.write_empty("source_compat", args.output)
        return 0

    native_cc = os.environ.get("NATIVE_CC", os.environ.get("CC", "gcc"))
    native_cxx = os.environ.get("NATIVE_CXX", os.environ.get("CXX", "g++"))
    loom_cc = compiler_path("LOOM_CC", ROOT / "build" / "bin" / "loom-cc")
    loom_cxx = compiler_path("LOOM_CXX", ROOT / "build" / "bin" / "loom-c++")

    rows: list[dict[str, str]] = []
    failed = False
    for case in cases:
        source_dir = ROOT / "test" / "app" / case
        if not (source_dir / "run_check.sh").is_file():
            rows.append(
                {
                    "case": case,
                    "suite": "app",
                    "native_status": "blocked",
                    "loom_status": "blocked",
                    "mode": "compatibility",
                    "diagnostic": "missing app run_check.sh",
                }
            )
            failed = True
            continue

        native_status, native_diag = run_case(source_dir, native_cc, native_cxx, "native")
        loom_status, loom_diag = run_case(source_dir, loom_cc, loom_cxx, "loom")
        if native_status == "pass" and loom_status == "pass":
            diagnostic = "native and loom drop-in runs passed"
        else:
            failed = True
            diagnostic = f"native: {native_diag}; loom: {loom_diag}"
        rows.append(
            {
                "case": case,
                "suite": "app",
                "native_status": native_status,
                "loom_status": loom_status,
                "mode": "compatibility",
                "diagnostic": diagnostic,
            }
        )

    write_rows(output, rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

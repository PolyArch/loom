#!/usr/bin/env python3
"""Emit dataflow primitive coverage summary rows."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


PRIMITIVES = (
    "stream",
    "carry",
    "invariant",
    "gate",
    "constant",
    "load",
    "store",
    "sync",
    "mux",
    "demux",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    return parser.parse_args(argv)


def discover_cases() -> list[str]:
    app_root = ROOT / "test" / "app"
    return sorted(path.name for path in app_root.iterdir() if (path / "dfg_check.sh").is_file())


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


def run_dfg_check(work_dir: Path) -> tuple[str, str]:
    env = os.environ.copy()
    env["LOOM_CC"] = tool_path("LOOM_CC", "loom-cc")
    env["LOOM_CXX"] = tool_path("LOOM_CXX", "loom-c++")
    env["LOOM_RAISE"] = tool_path("LOOM_RAISE", "loom-raise")
    env["LOOM_LOWER"] = tool_path("LOOM_LOWER", "loom-lower")
    env["LOOM_RAISE_OPT"] = tool_path("LOOM_RAISE_OPT", "loom-raise-opt")
    env["BUILD_DIR"] = str(work_dir / "build")
    result = subprocess.run(
        ["bash", str(work_dir / "dfg_check.sh")],
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return "pass", result.stdout.strip()
    detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return "fail", detail[0] if detail else f"dfg_check exited {result.returncode}"


def count_primitives(work_dir: Path) -> dict[str, int]:
    counts = {primitive: 0 for primitive in PRIMITIVES}
    pattern = re.compile(r"\bdataflow\.(" + "|".join(PRIMITIVES) + r")\b")
    for path in sorted((work_dir / "build").glob("*.dfg.mlir")):
        text = path.read_text()
        for match in pattern.finditer(text):
            counts[match.group(1)] += 1
    return counts


def rows_for_case(case: str) -> tuple[bool, list[dict[str, str]]]:
    source_dir = ROOT / "test" / "app" / case
    if not (source_dir / "dfg_check.sh").is_file():
        return True, [
            {
                "workload": case,
                "primitive": "none",
                "op_count": "0",
                "dfg_sim_status": "blocked",
                "diagnostic": "missing app dfg_check.sh",
            }
        ]

    temp_root = ROOT / "temp" / "test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"loom-primitive-{case}-", dir=temp_root) as tmp:
        work_dir = prepare_temp_app(source_dir, tmp)
        status, diagnostic = run_dfg_check(work_dir)
        if status != "pass":
            return True, [
                {
                    "workload": case,
                    "primitive": "none",
                    "op_count": "0",
                    "dfg_sim_status": "blocked",
                    "diagnostic": diagnostic,
                }
            ]
        counts = count_primitives(work_dir)

    rows: list[dict[str, str]] = []
    for primitive in PRIMITIVES:
        count = counts[primitive]
        if count > 0:
            diagnostic = "DFG-sim is not implemented; op-count coverage only"
        else:
            diagnostic = "primitive absent in generated dataflow; DFG-sim is not implemented"
        rows.append(
            {
                "workload": case,
                "primitive": primitive,
                "op_count": str(count),
                "dfg_sim_status": "blocked",
                "diagnostic": diagnostic,
            }
        )
    return False, rows


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cases = args.cases or discover_cases()
    if not cases:
        intermediate_artifacts.write_csv(
            "dataflow_primitive_coverage",
            intermediate_artifacts.output_path(args.output),
        )
        return 0

    all_rows: list[dict[str, str]] = []
    failed = False
    for case in cases:
        case_failed, rows = rows_for_case(case)
        failed = failed or case_failed
        all_rows.extend(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("dataflow_primitive_coverage", output, all_rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

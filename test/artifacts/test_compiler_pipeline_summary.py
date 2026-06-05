#!/usr/bin/env python3
"""Regression test for app compiler pipeline summary evidence."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import artifact_test_common


HEADER = [
    "case",
    "suite",
    "llvm_ir_status",
    "raised_mlir_status",
    "dataflow_status",
    "diagnostic",
]


def run_summary(repo: Path, output: Path, *args: str) -> list[dict[str, str]]:
    return artifact_test_common.run_csv_summary(
        repo,
        "test/app/run_compiler_pipeline_summary.sh",
        output,
        HEADER,
        *args,
        label="compiler pipeline summary",
    )


def assert_pass_row(row: dict[str, str], case: str) -> None:
    expected = {
        "case": case,
        "suite": "app",
        "llvm_ir_status": "pass",
        "raised_mlir_status": "pass",
        "dataflow_status": "pass",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"{case} {key}={row[key]!r}, expected {value!r}")
    if not row["diagnostic"].startswith("LLVM IR, raise, and dataflow checks passed"):
        raise AssertionError(f"unexpected diagnostic: {row['diagnostic']!r}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-compiler-pipeline-") as tmp:
        output = Path(tmp) / "compiler-pipeline-summary.csv"
        rows = run_summary(repo, output, "--case", "vecadd")
        vecadd_rows = [row for row in rows if row["case"] == "vecadd"]
        if len(vecadd_rows) != 1:
            raise AssertionError(f"expected one vecadd row, got {rows}")
        assert_pass_row(vecadd_rows[0], "vecadd")

        default_output = Path(tmp) / "compiler-pipeline-summary-default.csv"
        rows = run_summary(repo, default_output)
        expected_cases = {
            path.name
            for path in (repo / "test" / "app").iterdir()
            if (path / "raise_check.sh").is_file() and (path / "dfg_check.sh").is_file()
        }
        actual_cases = {row["case"] for row in rows}
        if expected_cases != actual_cases:
            raise AssertionError(f"default cases {actual_cases} do not match {expected_cases}")
        for row in rows:
            assert_pass_row(row, row["case"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

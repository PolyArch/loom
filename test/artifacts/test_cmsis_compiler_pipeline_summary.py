#!/usr/bin/env python3
"""Regression test for CMSIS compiler-pipeline summary evidence."""

from __future__ import annotations

import sys
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


def assert_pass_row(row: dict[str, str], case: str, suite: str) -> None:
    expected = {
        "case": case,
        "suite": suite,
        "llvm_ir_status": "pass",
        "raised_mlir_status": "pass",
        "dataflow_status": "pass",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"{case} {key}={row[key]!r}, expected {value!r}")
    if "IR, raise, and dataflow checks passed" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic for {case}: {row['diagnostic']!r}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-cmsis-pipeline-summary-") as tmp:
        output = Path(tmp) / "cmsis-compiler-pipeline-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/cmsis/run_compiler_pipeline_summary.sh",
                "--output",
                str(output),
            ],
            "CMSIS compiler pipeline summary",
        )
        rows = artifact_test_common.read_csv_rows(output, HEADER)
        by_case = {row["case"]: row for row in rows}
        if set(by_case) != {"cmsis-dsp", "cmsis-nn"}:
            raise AssertionError(f"unexpected CMSIS summary rows: {rows}")
        assert_pass_row(by_case["cmsis-dsp"], "cmsis-dsp", "CMSIS-DSP")
        assert_pass_row(by_case["cmsis-nn"], "cmsis-nn", "CMSIS-NN")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

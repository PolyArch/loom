#!/usr/bin/env python3
"""Regression test for app source compatibility summary evidence."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import artifact_test_common


HEADER = [
    "case",
    "suite",
    "native_status",
    "loom_status",
    "mode",
    "diagnostic",
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-source-compat-") as tmp:
        output = Path(tmp) / "source-compat-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/app/run_source_compat_summary.sh",
            output,
            HEADER,
            "--case",
            "vecadd",
            label="source compatibility summary",
        )

        vecadd_rows = [row for row in rows if row["case"] == "vecadd"]
        if len(vecadd_rows) != 1:
            raise AssertionError(f"expected one vecadd row, got {rows}")
        row = vecadd_rows[0]
        expected = {
            "suite": "app",
            "native_status": "pass",
            "loom_status": "pass",
            "mode": "compatibility",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f"vecadd {key}={row[key]!r}, expected {value!r}")
        if not row["diagnostic"].startswith("native and loom drop-in runs passed"):
            raise AssertionError(f"unexpected diagnostic: {row['diagnostic']!r}")

        default_output = Path(tmp) / "source-compat-summary-default.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/app/run_source_compat_summary.sh",
            default_output,
            HEADER,
            label="default source compatibility summary",
        )
        expected_cases = {
            path.name
            for path in (repo / "test" / "app").iterdir()
            if (path / "run_check.sh").is_file()
        }
        actual_cases = {row["case"] for row in rows}
        if expected_cases != actual_cases:
            raise AssertionError(f"default cases {actual_cases} do not match {expected_cases}")
        if any(row["native_status"] != "pass" or row["loom_status"] != "pass" for row in rows):
            raise AssertionError(f"default run should pass all current cases: {rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

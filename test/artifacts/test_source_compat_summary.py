#!/usr/bin/env python3
"""Regression test for app source compatibility summary evidence."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-source-compat-") as tmp:
        output = Path(tmp) / "source-compat-summary.csv"
        result = subprocess.run(
            [
                "bash",
                "test/app/run_source_compat_summary.sh",
                "--case",
                "vecadd",
                "--output",
                str(output),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"source compatibility summary failed with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        with output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        expected_header = [
            "case",
            "suite",
            "native_status",
            "loom_status",
            "mode",
            "diagnostic",
        ]
        if reader.fieldnames[: len(expected_header)] != expected_header:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")

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
        result = subprocess.run(
            [
                "bash",
                "test/app/run_source_compat_summary.sh",
                "--output",
                str(default_output),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"default source compatibility summary failed with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        with default_output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
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

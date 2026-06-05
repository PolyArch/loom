#!/usr/bin/env python3
"""Regression test for dataflow primitive coverage summary evidence."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


HEADER = ["workload", "primitive", "op_count", "dfg_sim_status", "diagnostic"]
EXPECTED_POSITIVE = {"stream", "carry", "invariant", "load"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames[: len(HEADER)] != HEADER:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")
        return rows


def run_summary(repo: Path, output: Path, *args: str) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "bash",
            "test/dataflow/run_primitive_coverage.sh",
            *args,
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
            f"primitive coverage summary failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return read_rows(output)


def assert_vecadd_rows(rows: list[dict[str, str]]) -> None:
    by_primitive = {row["primitive"]: row for row in rows if row["workload"] == "vecadd"}
    missing = sorted(EXPECTED_POSITIVE - set(by_primitive))
    if missing:
        raise AssertionError(f"missing vecadd primitive rows: {missing}; rows={rows}")
    for primitive in sorted(EXPECTED_POSITIVE):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"vecadd {primitive} count is not positive: {row}")
        if row["dfg_sim_status"] != "blocked":
            raise AssertionError(f"vecadd {primitive} simulator status should be blocked: {row}")
        if "DFG-sim is not implemented" not in row["diagnostic"]:
            raise AssertionError(f"unexpected diagnostic for {primitive}: {row}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-primitive-coverage-") as tmp:
        output = Path(tmp) / "dataflow-primitive-coverage.csv"
        assert_vecadd_rows(run_summary(repo, output, "--case", "vecadd"))

        default_output = Path(tmp) / "dataflow-primitive-coverage-default.csv"
        rows = run_summary(repo, default_output)
        expected_cases = {
            path.name
            for path in (repo / "test" / "app").iterdir()
            if (path / "dfg_check.sh").is_file()
        }
        actual_cases = {row["workload"] for row in rows}
        if expected_cases != actual_cases:
            raise AssertionError(f"default cases {actual_cases} do not match {expected_cases}")
        assert_vecadd_rows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
